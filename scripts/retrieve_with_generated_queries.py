import os
import sys
import datasets
import argparse
import pandas as pd
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import csv
import shutil
from pyserini.search.lucene import LuceneSearcher

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG
from src.retrieval.retrieve_bm25_monoT5 import Retriever
from src.models.monoT5 import MonoT5
from costum_dataset import collate_fn,CustomDataset
from src.retrieval.query_aggregation import (
    sort_all_scores,
    rerank_against_query,
    vote,
    query_filter,
    simple_retrieval,
    combSum,
)


def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", type=str, default="segmented", 
        choices=["complete","segmented"]
    )
    parser.add_argument(
        "--output_file", type=str, 
    )
    args = parser.parse_args()

    queries_file = CONFIG["retrieval"]["generated_queries_file"]
    print("Loading generated queries from :", queries_file)
    gen_queries = pd.read_csv(
        queries_file, converters={"generated_text": eval}, index_col=0
    )
    gen_queries = gen_queries.rename(
        columns={"generated_text": "generated_queries"}
    )

    #### filtering queries
    if CONFIG["retrieval"]["filter_queries"]:
        print("Filtering queries")
        gen_queries["generated_queries"] = gen_queries.apply(
            lambda x: query_filter(x["query"], x["generated_queries"]), axis=1
        )
    else:
        gen_queries["generated_queries"] = gen_queries[
            "generated_queries"
        ].apply(lambda x: [q[1:] for q in x if q[0] == " "])

    ranker = Retriever(corpus = CONFIG['corpus'][args.corpus], index = args.corpus)
    monot5 = MonoT5(args.corpus, device="cuda")

    results = []

    print("Retrieval")
    aggregation = CONFIG["retrieval"]["query_aggregation"]
    dataset = gen_queries

    file_out =  [['topic_id','fixed', 'docid' , 'score', 'run_id']]

    for i,row in tqdm(dataset.iterrows()) : 
        query_id = row["query_id"]
        query = row["query"]
        if aggregation == "sort":
            top_docs = sort_all_scores(query, row["generated_queries"], ranker)
        elif aggregation == "rerank":
            top_docs = rerank_against_query(
                query, row["generated_queries"], ranker, monot5
            )
        elif aggregation == "combSum":
            searcher = LuceneSearcher(CONFIG['corpus'][args.corpus])
            top_docs = combSum(
                query, row["generated_queries"], ranker, searcher, MNZ=False
            )
        elif aggregation == "combMNZ":
            searcher = LuceneSearcher(CONFIG['corpus'][args.corpus])
            top_docs = combSum(
                query, row["generated_queries"], ranker, searcher, MNZ=True
            )
        elif aggregation == "vote":
            searcher = LuceneSearcher(CONFIG['corpus'][args.corpus])
            top_docs = vote(query, row["generated_queries"], ranker,searcher)
        elif aggregation == "summed_vote":
            searcher = LuceneSearcher(CONFIG['corpus'][args.corpus])
            top_docs = vote(
                query,
                row["generated_queries"],
                ranker,
                score_compute="sum",
            )
        elif aggregation == "mean_vote":
            searcher = LuceneSearcher(CONFIG['corpus'][args.corpus])
            top_docs = vote(
                query,
                row["generated_queries"],
                ranker,
                score_compute="mean",
            )
        elif aggregation == "seperate_queries":
            for q in row["generated_queries"]:
                ids, text, score = simple_retrieval(q, ranker)
                results.append(
                    {
                        "query": query,
                        "query_id": query_id,
                        "sub_query": q,
                        "retrieved_ids": ids,
                        "retrieved_passages": [],
                        "score": score,
                        #"reference_ids": [q["docid"] for q in row["quotes"]],
                        #"answers": row["answers"],
                        #"gold_quotes": row["quotes"],
                    }
                )
                ids, text, score = simple_retrieval(query, ranker)

            # je ne dois pas sauter?

        results.append(
                {
                    "query":{
                        "id" : query_id,
                        "text" : query
                    },
                    
                    "candidates": top_docs
                }
            )

        i = 1  
        for d in top_docs :
            # Construire une liste de dictionnaires à ajouter
            file_out.append([query_id, 'Q0', d['docid'], i, d['score'], 'ISIR'])
            i+=1

    end = time.time()
    print("time: ", end - start)
        
    results_df = pd.DataFrame.from_dict(results)

    experiment_folder = (
        CONFIG["retrieval"]["experiment_path"] + CONFIG["retrieval"]["experiment_name"]
    )
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
        print("New directory for experiment is created: ", experiment_folder)

    # Création et écriture des données dans le fichier TSV
    with open(os.path.join(experiment_folder, args.output_file), mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(file_out)

    results_df.to_csv(
        f"{experiment_folder}/{CONFIG['retrieval']['query_gen_results_file']}"
    )
    #file_out.to_csv(f"{experiment_folder}/{args.output_file}", sep='\t', index=False)
    print(
        "Results saved in",
        f"{experiment_folder}/{CONFIG['retrieval']['query_gen_results_file']}",
    )

if __name__ == "__main__":

    main()

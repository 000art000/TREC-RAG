import os
import sys
import torch
import pandas as pd
import datasets
from torch.utils.data import DataLoader
import os
import shutil
import time
from tqdm import tqdm
import argparse
import csv
import re

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG
from src.retrieval.retrieve_bm25_monoT5 import Retriever
from costum_dataset import CustomDataset, collate_fn


def main():
    start = time.time()

    torch.set_grad_enabled(False)

    results = []
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_set", type=str, default="MS_MARCO_Dev2", 
        choices=["MS_MARCO_Dev2", "MS_MARCO_Dev", "TREC_DL_2023", "TREC_DL_2022", "TREC_DL_2021", 'comp24']
    )
    parser.add_argument(
        "--output_file", type=str, 
    )
    parser.add_argument(
        "--corpus", type=str, default="segmented", 
        choices=["complete","segmented"]
    )

    args = parser.parse_args()
    dataset = CONFIG["dataset"][args.data_set]

    c = CustomDataset( f"{CONFIG['data_path']}/{dataset}",1)
    dataset = DataLoader(c,  batch_size=128, collate_fn=collate_fn)
    ranker = Retriever(corpus=CONFIG['corpus'][args.corpus], index = args.corpus)

    file_out = [['topic_id','fixed', 'docid' , 'score', 'run_id']]
    for batch in tqdm(dataset) : 
        for row in batch:
            query_id = row["query_id"]
            query = row["query"]

            top_docs = ranker.search(query,k=100)
            
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
                file_out.append([ query_id, 'Q0', d['docid'], i, d['score'], 'ISIR'])
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
        
    results_df.to_csv(f"{experiment_folder}/{CONFIG['retrieval']['results_file']}")
    #file_out.to_csv(f"{experiment_folder}/{args.output_file}", sep='\t', index=False)
    print(
        "Results saved in", f"{experiment_folder}/{CONFIG['retrieval']['results_file']}"
    )


if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
import pandas as pd
import argparse
from tabulate import tabulate
import json


ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG

from src.evaluation.generation_metrics import *
from src.data.hagrid_dataset_tools import get_attributable_answer, get_all_answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architcture",
        type=str,
        default="G",
        choices=["G", "RTG-gold", "RTG-vanilla", "RTG-query-gen"],
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    experiment = CONFIG["architectures"][args.architcture]

    results_folder = experiment["experiment_path"] + experiment["experiment_name"]

    results_file = args.results_file if args.results_file else (  results_folder + "/" + experiment["results_file"])
    print("Loading results file:", results_file)
    generated_column = CONFIG["column_names"]["prediction"]
    reference_column = CONFIG["column_names"]["reference"]
    
    if results_file.endswith(".json"):
        with open(results_file) as f:
            json_dict = json.load(f)
            results = pd.json_normalize(json_dict["data"])
    elif CONFIG["dataset"] == "HAGRID":
        results = pd.read_csv(
            results_file, index_col=[0], converters={reference_column: eval}
        )
    else:
        results = pd.read_csv(results_file, index_col=[0])

    ## processing the generated text to remove system prompt

    pattern = r"<\|system\|>[\s\S]*?<\|assistant\|>\n"
    results["processed_generated_text"] = results.apply(
        lambda x: x[generated_column].replace("<|endoftext|>", ""), axis=1
    )
    results["processed_generated_text"] = results[
        "processed_generated_text"
    ].str.replace(pattern, "", regex=True)
    print("example processed text : ", results["processed_generated_text"][0], "\n")

    if CONFIG["dataset"] == "HAGRID":
        results["gold_answer"] = results[reference_column].apply(get_attributable_answer)
        results = results[results["gold_answer"].str.len() > 0]
    else:
        results["gold_answer"] = results[reference_column]


    ### remove citiations from answer
    citation_pattern = r"\[\d+(?:,\s*\d+)*\]"
    results["gold_answer"] = results["gold_answer"].str.replace(
        citation_pattern, "", regex=True
    )
    if CONFIG["dataset"] == "HAGRID":
        # add all answer without citiations
        results["all_gold_answer"] = results[reference_column].apply(get_all_answers)
        print(
            "example all gold answer without citations: ",
            results["all_gold_answer"][0],
            "\n",
        )

    print("example gold answer without citations: ", results["gold_answer"][0], "\n")

    if experiment["citation"]:
        results["processed_generated_text"] = results[
            "processed_generated_text"
        ].str.replace(citation_pattern, "", regex=True)
        print(
            "example generated answer without citations : ",
            results["processed_generated_text"][0],
            "\n",
        )

    reference_column = "gold_answer"
    generated_column = "processed_generated_text"

    # with only one reference answer
    print("loading rouge metric")
    rouge_scores = rouge(
        results[generated_column],
        results[reference_column],
        CONFIG,
        use_aggregator=False,
        rouge_types=["rougeLsum"],
    )  # new Huggingface version of rouge which returns one aggregated value
    print("loading rouge metric dataset version ")
    rouge_scores_ov = rouge_detailed_ov(
        results[generated_column], results[reference_column], CONFIG
    )  # Huggingface old version of rouge that details precision/recall/fscore
    print("rouge score:", rouge_scores)
    print("loading bert metric")
    bert_scores = bert_score(
        results[generated_column], results[reference_column], CONFIG
    )
    # bleu score
    print("load bleu score")
    bleu_score = bleu(results[generated_column], results[reference_column], CONFIG)
    rouge_scores_ov_all = None
    bert_scores_all = None
    if CONFIG["dataset"] == "HAGRID":
        # with all reference answer

        reference_column = "all_gold_answer"

        print("loading rouge metric dataset version, all")
        rouge_scores_ov_all = rouge_detailed_ov_all(
            results[generated_column], results[reference_column], CONFIG
        )

        print("loading bert metric, all")
        bert_scores_all = bert_metric_all(
            results[generated_column], results[reference_column], CONFIG
        )

    performance = {
        "Rouge": {
            "Precision": rouge_scores_ov["rougeLsum"][1][0],
            "Recall": rouge_scores_ov["rougeLsum"][1][1],
            "fmeasure": rouge_scores_ov["rougeLsum"][1][2],
        },
        "Bert": {
            "Precision": np.array(bert_scores["precision"]).mean(),
            "Recall": np.array(bert_scores["recall"]).mean(),
            "fmeasure": np.array(bert_scores["f1"]).mean(),
        },
        "Aggregated Rouge": {
            "Precision": rouge_scores["rougeLsum"],
            "Recall": rouge_scores["rougeLsum"],
            "fmeasure": rouge_scores["rougeLsum"],
        },
        "Rouge All": {
            "Precision": np.mean(rouge_scores_ov_all["precision"]),
            "Recall": np.mean(rouge_scores_ov_all["recall"]),
            "fmeasure": np.mean(rouge_scores_ov_all["fmeasure"]),
        },
        "Bert All": {
            "Precision": np.mean(bert_scores_all["precision"]),
            "Recall": np.mean(bert_scores_all["recall"]),
            "fmeasure": np.mean(bert_scores_all["f1"]),
        },
    }

    print(experiment["experiment_name"])

    performance_df = pd.DataFrame(performance).T
    print(f"Aggregated metrics for the complete dataset")
    print(tabulate(performance_df, headers="keys", tablefmt="presto"))
    performance_df.to_csv(os.path.join(results_folder, "performance_bert.csv"))
    bleu_score_all = None
    if CONFIG["dataset"] == "HAGRID":
        print("load bleu score all")
        bleu_score_all = bleu_all(
            results[generated_column], results[reference_column], CONFIG
        )

    performance_bleu = {
        "Bleu ": {
            "Precision": bleu_score["bleu"],
            "1g": bleu_score["precisions"][0],
            "2g": bleu_score["precisions"][1],
            "3g": bleu_score["precisions"][2],
            "4g": bleu_score["precisions"][3],
        },
        "Bleu All": {
            "Precision": bleu_score_all["bleu"],
            "1g": bleu_score_all["precisions"][0],
            "2g": bleu_score_all["precisions"][1],
            "3g": bleu_score_all["precisions"][2],
            "4g": bleu_score_all["precisions"][3],
        },
    }

    performance_df = pd.DataFrame(performance_bleu).T
    print(f"Aggregated metrics for the complete dataset")
    print(tabulate(performance_df, headers="keys", tablefmt="presto"))

    performance = performance_bleu | performance

    results_file = results_file.replace(".csv", "_perf_answer.json")

    with open(results_file, "w") as f:
        json.dump(performance, f, indent=4)

if __name__ == "__main__":
    main()

import os
import sys
import argparse
import json
import time

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from transformers import set_seed
from torch.utils.data import DataLoader
import re 
ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG

from src.generation.llms.zephyr import generate_queries
from costum_dataset import collate_fn, CustomDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="zephyr", choices=["zephyr", "llama2"]
    )
    parser.add_argument(
        "--data_set", type=str, default="MS_MARCO_Dev2", 
        choices=["MS_MARCO_Dev2", "MS_MARCO_Dev", "TREC_DL_2023", "TREC_DL_2022", "TREC_DL_2021", "comp24"]
    )
    
    args = parser.parse_args()
    model_config = CONFIG["langauge_model"][args.model_name]
    dataset = CONFIG["dataset"][args.data_set]

    set_seed(model_config["SEED"])
    exception = False
    results = None
    execution_time = 0
    try:
        model_id = model_config["model_id"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir = model_config["cache_dir"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=model_config["cache_dir"],
            trust_remote_code=True,
            device_map="auto",
        )

        c = CustomDataset( f"{CONFIG['data_path']}/{dataset}",1)
        dataset = DataLoader(c,  batch_size=128, collate_fn=collate_fn)

        results = []
        prompt = CONFIG["prompts"]["query_gen_prompt"]
        start = time.time()

        for i,batch in tqdm(enumerate(dataset)):
            
            for row in batch :
                #answer a supprimer
                answer = None
                if CONFIG["query_generation"]["include_answer"]:
                    answer = row["answers"][0]["answer"]
                examples = None
                if CONFIG["query_generation"]["setting"] == "fewshot":
                    examples = CONFIG["query_generation"]["fewshot_examples"]
                nb_queries_to_generate = CONFIG["query_generation"][
                    "nb_queries_to_generate"
                ]
                nb_shots = CONFIG["query_generation"]["nb_shots"]
                queries = generate_queries(
                    row["query"],
                    model,
                    tokenizer,
                    prompt,
                    include_answer=CONFIG["query_generation"]["include_answer"],
                    answer=answer,
                    fewshot_examples=examples,
                    nb_queries_to_generate=nb_queries_to_generate,
                    nb_shots=nb_shots,
                )
                results.append(
                    {
                        "query": row["query"],
                        "query_id": row["query_id"],
                        "generated_text": queries,
                    }
                )
        end = time.time()

        execution_time = (end - start) / 60
    except:
        exception = True
        print("Exception caught")
        raise
    finally:
        print("Saving experiment")

        experiment_folder = (
            CONFIG["query_generation"]["experiment_path"]
            + args.model_name + CONFIG["query_generation"]["experiment_name"]
        )
        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)
            print("New directory for experiment is created: ", experiment_folder)
        if results is not None:
            res = re.sub('(\d+)shot', f"{CONFIG['query_generation']['nb_shots']}shot", CONFIG['query_generation']['results_file'])
            res = re.sub('(\d+)q', f"{CONFIG['query_generation']['nb_queries_to_generate']}q", CONFIG['query_generation']['results_file']) 
            exp_config = CONFIG['architectures']["RTG-query-gen"]
            results_df = pd.DataFrame.from_dict(results)
            results_df.to_csv(
                f"{experiment_folder}/{res}"
            )
            print(
                "Result file:",
                f"{experiment_folder}/{res}",
            )
            config_file = (
                f"{experiment_folder}/{re.sub('.csv','.json',res)}"
            )
            exp_config["execution_time"] = str(execution_time) + " minutes"
            exp_config["error"] = exception
            with open(config_file, "w") as file:
                json.dump(exp_config, file)
        torch.cuda.empty_cache()


main()

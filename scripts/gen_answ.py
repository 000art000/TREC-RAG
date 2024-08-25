import os
import sys
import re
import json
import time
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import datasets
from transformers import set_seed
from torch.utils.data import DataLoader
from peft import prepare_model_for_kbit_training
import csv
from pyserini.search.lucene import LuceneSearcher
from nltk import sent_tokenize
import unicodedata

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG
from src.data.data import prepare_contexts, remove_special_spaces, get_passage, extract_sentences_and_citations, format_support_passages
from costum_dataset import CustomDataset, collate_fn

def main():

    # Initial parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="zephyr", choices=["zephyr", "llama2","llama3", 't5_large', 't5_xl' ]
    )
    parser.add_argument(
        "--architcture",
        type=str,
        default="G",
        choices=["G", "RTG-gold", "RTG-vanilla", "RTG-query-gen"],
    )

    parser.add_argument(
        "--load",
        type=str,
    )
    parser.add_argument(
        "--output_file", type=str, 
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
    )
    parser.add_argument(
            "--data_set", type=str, default="MS_MARCO_Dev2", 
            choices=["MS_MARCO_Dev2", "MS_MARCO_Dev", "TREC_DL_2023", "TREC_DL_2022", "TREC_DL_2021", 'comp24']
        )
    parser.add_argument(
            "--input_file", type=str
    )
    parser.add_argument(
        "--corpus", type=str, default="segmented", 
        choices=["complete","segmented"]
    )

    # Parsing initial arguments
    args = parser.parse_args()

    # Load configurations based on initial arguments
    model_config = CONFIG["langauge_model"][args.model_name]
    experiment = CONFIG["architectures"][args.architcture]

    #vanilla = args.architcture == "RTG-vanilla"

    torch.set_grad_enabled(False)

    dataset = CONFIG["dataset"][args.data_set]
    c = CustomDataset(f"{CONFIG['data_path']}/{dataset}", 1)
    dataset = DataLoader(c, batch_size=128, collate_fn=collate_fn)

    set_seed(model_config["SEED"])
    exception = False
    results = None
    execution_time = 0
    nb_passages = None
    use_support_doc = experiment["use_context"]

    try:

        if args.load :
            model_save_path = 'results/' + args.load
            model = AutoModelForCausalLM.from_pretrained(model_save_path)
            if args.model_name == "llama3" or args.model_name == "llama2":
                model_id = model_config[args.model_name]['model_id']
            elif args.model_name == "t5_large" or args.model_name == "t5_xl" : 
                model_id = model_config[args.model_name]['model_id']
            else :
                model_id = model_config["model_id"]

        else :
            if args.model_name == "llama3" or args.model_name == "llama2":
                model_id = model_config[args.model_name]['model_id']
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    device_map="auto",
                )
            elif args.model_name == "t5_large" or args.model_name == "t5_xl" : 
                model_id = model_config[args.model_name]['model_id']
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    device_map="auto"
                )
            else :
                model_id = model_config["model_id"]
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    device_map="auto",
                )
            
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=model_config["cache_dir"],
            model_max_length = model_config['max_input_length']
        )
        

        #model = prepare_model_for_kbit_training(model)

        model.eval()

        if experiment["use_retrieved"]:
            #path_passage =  CONFIG['retrieval']['retreival_dir_result'] +'/'+ args.input_dir
            path_passage = os.path.join(CONFIG['retrieval']['experiment_path']+ '/'+ CONFIG['retrieval']['experiment_name'], args.input_file)
            searcher = LuceneSearcher.from_prebuilt_index(CONFIG['corpus'][args.corpus])
            #dataset = datasets.Dataset.from_pandas(dataset_passages)
        nb_passages = experiment["nb_passages"]

        results = []
        if use_support_doc:
            searcher = LuceneSearcher.from_prebuilt_index(CONFIG['corpus'][args.corpus])
            if experiment["citation"]:
                prompt = CONFIG["prompts"]["prompt_10"]
            else:
                prompt = CONFIG["prompts"]["prompt_without_citation"]
        else:
            prompt = CONFIG["prompts"]["prompot_without_context"]
        start = time.time()
        ## loading data

        file_out = pd.DataFrame(columns=['run_id', 'topic_id', 'topic', 'references', 'response_length', 'answer'])
        passage = None
        for batch in tqdm(dataset):
            for row in batch :
                if experiment["use_retrieved"] :
                    passage = get_passage(row['query_id'], path_passage, CONFIG)
                file_out = aux(row, use_support_doc, args.corpus, nb_passages, passage ,experiment, prompt, tokenizer, model, model_config, file_out, searcher, args)

        end = time.time()
        execution_time = (end - start) / 60  
    except Exception as e:
        print(e)
        exception = True
        print("Exception caught")
        raise
    finally:
        print("Saving experiment")

        experiment_folder = (
            experiment["experiment_path"] + experiment["experiment_name"]
        )
        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)
            print("New directory for experiment is created: ", experiment_folder)
        if results is not None:
            exp_config = experiment
            #results_df = pd.DataFrame.from_dict(results)
            #results_file = args.results_file if args.results_file else experiment['results_file']
            #results_df.to_csv(f"{experiment_folder}/{results_file}",index = False)
            file_out.to_json(f"{experiment_folder}/{args.output_file}", orient='records', lines=True)

            config_file = f"{experiment_folder}/{experiment['config_file']}"
            exp_config["execution_time"] = str(execution_time) + " minutes"
            exp_config["error"] = exception
            with open(config_file, "w") as file:
                json.dump(exp_config, file)

############################################################################

def split_text_by_citations(text):
    # Define a regex pattern to match the citations in various formats
    citation_pattern = r'\[\d+(?:, \d+)*\](?:\[\d+\])*'
    initial_citations_pattern = r'^(' + citation_pattern + r'\s*)+'
    number_pattern = re.compile(r"\d+")
    
    # Split the text into sentences
    sentences = sent_tokenize(text)
    
    statements = []
    sent_without_src = ''

    for sentence in sentences:
        sentence = sentence.strip()
        src = re.findall(citation_pattern, sentence)

        initial_citations_match = re.match(initial_citations_pattern, sentence) 
        if initial_citations_match:
            # If the sentence starts with a citation, append the source to the previous sentence
            initial_citations_text = initial_citations_match.group(0)
            initial_citations = re.findall(citation_pattern, initial_citations_text)
            initial_int_src = list(set([int(e) for e in number_pattern.findall(" ".join(initial_citations))]))

            # Remove the initial citations from the sentence
            sentence = sentence[len(initial_citations_text):].strip()

            if sent_without_src:
                statements.append({"text":sent_without_src.strip(), "citations": initial_int_src})
                sent_without_src = ""
            elif statements:
                statements[-1]["citations"].extend(initial_int_src)

        int_src = list(set([int(e) for e in number_pattern.findall(" ".join(src))]))

        # Remove all citations from the sentence before adding it to the text
        for s in src:
            sentence = sentence.replace(s, "").strip()

        if len(int_src):
            sent_without_src += ' ' + sentence
            statements.append({"text":sent_without_src.strip(), "citations": int_src})
            sent_without_src = ""
        else:
            sent_without_src += ' ' + sentence
    
    # Add the last statement
    if sent_without_src.strip():
        statements.append({"text":sent_without_src.strip(), "citations": int_src})
    
    return statements

def filter_statements(statements, nb_doc):
    filtered_statements = []
    for i,statement in enumerate(statements):
        # Filtrer les sources pour ne garder que les numéros <= nb_doc
        filtered_sources = [num - 1 for num in statement['citations'] if num <= nb_doc]

        # Si la liste filtrée n'est pas vide, on met à jour la source et on garde le statement
        statement['citations'] = filtered_sources
        if i == 0 :
            filtered_statements.append(statement)
        else :
            if not filtered_sources:
                if filtered_statements and not filtered_statements[-1]['citations']: 
                    filtered_statements[-1]['text'] = filtered_statements[-1]['text'] + ' ' + statement['text']
                    filtered_statements[-1]['citations'] = []
                else:
                    filtered_statements.append(statement)
            else:
                filtered_statements.append(statement)

    return filtered_statements


def aux(row, use_support_doc, corpus, nb_passages, passage ,experiment, prompt, tokenizer, model, model_config, file_out, searcher, args):

    row[CONFIG["column_names"]["query"]] = remove_special_spaces(row[CONFIG["column_names"]["query"]])
    user_prompt = re.sub(r"\{query\}", row[CONFIG["column_names"]["query"]], prompt["user"])

    #passages = dataset_passages[pd.Series(dataset_passages['query_id']) == row['query_id']]

    if use_support_doc:
        context = prepare_contexts(
            passage[:nb_passages],
            CONFIG, 
            corpus,
            searcher,
            citation = experiment["citation"]
        )
        user_prompt = re.sub("\{context\}", re.escape(context), user_prompt)
    input_text = [
        {
            "role": "system",
            "content": prompt["system"],
        },
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        input_text,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if args.model_name == 'zephyr' :
        tokens = model.generate(
            inputs.to(model.device),
            max_new_tokens=model_config["max_new_tokens"],
            temperature=model_config["temperature"],
            pad_token_id=tokenizer.eos_token_id,
        )
    elif args.model_name == 't5_large' or args.model_name == 't5_xl' :
        tokens = model.generate(
            inputs.to(model.device),
            max_new_tokens=model_config["max_new_tokens"],
        )
    elif args.model_name == 'llama2' or args.model_name == 'llama3' :
        tokens = model.generate(
            inputs.to(model.device),
            max_new_tokens=model_config["max_new_tokens"],
            temperature=model_config["temperature"],
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(tokens[0], skip_special_tokens=True)
    filtered_answer = None

    if args.model_name == 'llama2' :
        matches = re.findall(r'ANSWER:\s*(.*)', answer)

        # Récupérer la dernière occurrence
        if matches:
            filtered_answer = matches[-1]
            filtered_answer = filtered_answer.strip()

    elif args.model_name == 'zephyr':
        pattern = r"<\|system\|>[\s\S]*?<\|assistant\|>\n"
        filtered_answer = answer.replace("<|endoftext|>", "")
        filtered_answer = re.sub(pattern, "", filtered_answer)

        filtered_answer = re.sub(r'QUESTION:.*', '', filtered_answer, flags=re.DOTALL)
        filtered_answer = filtered_answer.strip()
    else:
        pattern = r"<\|system\|>[\s\S]*?<\|assistant\|>\n"
        filtered_answer = answer.replace("<|endoftext|>", "")
        filtered_answer = re.sub(pattern, "", filtered_answer)

        filtered_answer = re.sub(r'QUESTION:.*', '', filtered_answer, flags=re.DOTALL)
        filtered_answer = filtered_answer.strip()

    if experiment["use_retrieved"]:
        passages = format_support_passages(passage[:nb_passages], searcher, CONFIG)
        answer = filter_statements(split_text_by_citations(filtered_answer),nb_passages)
        length = 0
        for sent in answer:
            text = sent['text'].strip()
            tokenized = unicodedata.normalize('NFKC', text)
            tokens = tokenized.split()
            length += len(tokens)

        # Construire une liste de dictionnaires à ajouter
        data_to_append = [{
            'run_id': f'ISIR-IRIT-{args.model_name}_p{nb_passages}', 
            'topic_id': row['query_id'], 
            'topic': row['query'],
            'references': passages, 
            'response_length': length, 
            'answer': answer
        }]

    else:

        # Construire une liste de dictionnaires à ajouter
        data_to_append = [{
            'run_id': 'ISIR', 
            'topic_id': row['query_id'], 
            'topic': row['query_id'],
            'references': None, 
            'response_length': len(filtered_answer), 
            'answer': filtered_answer
        }]

    file_out = pd.concat([file_out, pd.DataFrame(data_to_append)], ignore_index=True)
    torch.cuda.empty_cache()
    return file_out

main()

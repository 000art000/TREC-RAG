from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import torch 
import argparse, sys, os
from torch.utils.data import DataLoader
import re 
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)
from config import CONFIG
from costum_dataset import CustomDataset, collate_fn
from src.data.data import prepare_contexts, remove_special_spaces, get_passage, extract_sentences_and_citations, format_support_passages

model_id = CONFIG['langauge_model']['t5_large']['t5_large']['model_id'] 

# Load the tokenizer and model with the authentication token
tokenizer = AutoTokenizer.from_pretrained(
    model_id, use_auth_token=True, device_map="auto", cache_dir=CONFIG['langauge_model']['t5_large']['cache_dir']
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id, use_auth_token=True, device_map="auto", cache_dir=CONFIG['langauge_model']['t5_large']['cache_dir']
)

# Update the generator pipeline
generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device_map="auto", max_new_tokens=CONFIG['langauge_model']['t5_large']["max_new_tokens"]
)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_set", type=str, default="MS_MARCO_Dev2", 
    choices=["MS_MARCO_Dev2", "MS_MARCO_Dev", "TREC_DL_2023", "TREC_DL_2022", "TREC_DL_2021", 'comp24']
)
parser.add_argument(
    "--corpus", type=str, default="segmented", 
    choices=["complete", "segmented"]
)
parser.add_argument(
    "--input_file", type=str, 
)
args = parser.parse_args()

# Data loading and processing
dataset = CONFIG["dataset"][args.data_set]
c = CustomDataset(f"{CONFIG['data_path']}/{dataset}", 1)
dataset = DataLoader(c, batch_size=1, collate_fn=collate_fn)

# Set up searcher and input file path
path_passage = os.path.join(CONFIG['retrieval']['experiment_path'] + CONFIG['retrieval']['experiment_name'], args.input_file)
searcher = LuceneSearcher.from_prebuilt_index(CONFIG['corpus'][args.corpus])
text_id = "body" if args.corpus == 'complete' else "segment"
pos_docid = CONFIG['passage_column']['docid']

# Main loop for processing data
for batch in tqdm(dataset):
    for row in batch:
        query_id = row["query_id"]
        query = row["query"]

        docs_id = get_passage(query_id, path_passage, CONFIG)
        text_doc = ''

        for i in range(len(docs_id)):
            doc = eval(searcher.doc(docs_id[i][pos_docid]).raw())
            text_doc += f'id_{i}: {doc[text_id]} \n '

        prompt = CONFIG['prompt_rerank']
        prompt = re.sub("QUERY", query, prompt)
        prompt = re.sub("DOCS", re.escape(text_doc), prompt)
        
        # Generate output without model_max_length
        output = generator(prompt)
        for answer in output :

            filtered_answer = re.sub(r'.*ANSWER:\s*', '', output, flags=re.DOTALL)

        print(filtered_answer)

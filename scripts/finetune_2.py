import argparse
import re
import torch
import sys
import os
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, AdamW, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG
from src.evaluation.generation_metrics import *

class WebGPTDataset(Dataset):
    def __init__(self, train=True, test_size=0.2, random_state=0):
        # Charger le dataset train
        dataset = load_dataset("openai/webgpt_comparisons")['train']
        
        # Obtenir les indices pour les ensembles de formation et de test
        train_indices, test_indices = train_test_split(
            list(range(len(dataset))), test_size=test_size, random_state=random_state
        )
        
        # Créer des subsets en utilisant les indices
        if train:
            self.dataset = Subset(dataset, train_indices)
        else:
            self.dataset = Subset(dataset, test_indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def collate_fn(batch):
    question_batch = []
    answer_batch = []

    for item in batch:
        if item['score_0']>=item['score_1'] :
            q,a = item['quotes_0']['extract'], item['answer_0'] 
            #collated_batch.append({'question':item['question']['full_text'], 'quotes' : item['quotes_0']['extract'], 'answer': item['answer_0']  })
        else :
            q,a = item['quotes_1']['extract'], item['answer_1'] 
            #collated_batch.append({'question':item['question']['full_text'], 'quotes' : item['quotes_1']['extract'], 'answer': item['answer_1']  })


        prompt = CONFIG["prompts"]["prompt_1"]

        user_prompt = re.sub(r"\{query\}", re.escape(item["question"]['full_text']), prompt["user"])

        context = prepare_contexts(
            q,
            citation = True
        )
        """user_prompt = re.sub("\{context\}", re.escape(context), user_prompt)

        input_text = [
            {
                "role": "system",
                "content": prompt["system"],
            },
            {"role": "user", "content": user_prompt},
        ]"""

        user_prompt = re.sub(r"\{context\}", re.escape(context), user_prompt)
        input_text = prompt["system"] + " " + user_prompt
      
        question_batch.append(input_text)

        reponse = prompt["system"] + " " + user_prompt + " " + a
        answer_batch.append(reponse)

    return question_batch, answer_batch

def evaluate(model, eval_loader, tokenizer):
    model.eval()
    score = 0
    nb =0

    with torch.no_grad():
        for i,(questions, answers) in tqdm( enumerate(eval_loader)):
            inputs = tokenizer(questions, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits

            # Convertir les logits en identifiants de tokens
            token_ids = torch.argmax(logits, dim=-1)
            # Décoder les identifiants en texte
            responses = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            s= bert_score(responses, answers,CONFIG)['f1']
            score+= np.sum(s)
            nb+= len(s)


    m = score/nb
    return m

Load = False
model_name = "t5_large"

def main():
    train_dataset = WebGPTDataset()
    train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

    model_config = CONFIG["langauge_model"][model_name]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetoire", type=str)
    args = parser.parse_args()

    try:

        if not Load :

            # Créer une configuration BitsAndBytesConfig pour la quantification
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

            model_id = None

            if model_name == "llama3" or model_name == "llama2":
                model_id = model_config[model_name]['model_id']
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    device_map="auto",
                    quantization_config=quantization_config
                )
            elif model_name == "t5_large" or model_name == "t5_xl" : 
                model_id = model_config[model_name]['model_id']
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    device_map="auto",
                    quantization_config=quantization_config
                )
            else :
                model_id = model_config["model_id"]
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    device_map="auto",
                    quantization_config=quantization_config
                )

            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=model_config["cache_dir"],
                model_max_length = model_config['max_input_length']
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )

            # Geler les poids du modèle principal pour désactiver leurs gradients
            for param in model.parameters():
                param.requires_grad = False

            model = get_peft_model(model, peft_config)
            print_trainable_parameters(model)
        
        else :
            model_save_path = 'results/' + args.repetoire
            tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            model = AutoModelForCausalLM.from_pretrained(model_save_path)

            # Réactiver gradient_checkpointing
            model.gradient_checkpointing_enable()

            # Réactiver enable_input_require_grads
            model.enable_input_require_grads()

            for name, param in model.named_parameters():
                if "lora" in name:  # Si le nom du paramètre contient "lora", activez le gradient
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            print_trainable_parameters(model)

        training_arguments = TrainingArguments(
            output_dir="./results",
            num_train_epochs=5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            optim="adamw_torch",
            save_steps=25,
            logging_steps=5,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=False,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="all",
            evaluation_strategy="steps",
            eval_steps=5,
        )

        optimizer = AdamW(model.parameters(), lr=training_arguments.learning_rate, weight_decay=training_arguments.weight_decay)
        best_eval_score = 0.9079744246872988

        # Create the directory if it does not exist
        model_save_path = 'results/' + args.repetoire
        os.makedirs(model_save_path, exist_ok=True)

        for epoch in range(training_arguments.num_train_epochs):
            model.train()
            for questions, answers in tqdm(train_loader):
                optimizer.zero_grad()

                inputs = tokenizer(questions, return_tensors='pt', padding=True, truncation=True)
                labels = tokenizer(answers, return_tensors='pt', padding=True, truncation=True)

                max_length_labels = labels['input_ids'].shape[1]

                inputs = tokenizer(questions, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length_labels)
                labels = tokenizer(answers, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length_labels)

                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                labels = labels['input_ids'].to(model.device)

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()

                optimizer.step()

            valid_dataset = WebGPTDataset(train=False)
            valid_loader = DataLoader(valid_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

            sys.stdout = open(os.devnull, 'w')
            eval_results = evaluate(model, valid_loader, tokenizer)
            current_score = eval_results # Use f1_score or another metric you prefer
            sys.stdout = sys.__stdout__
            print(f'\ncurrent_score {current_score} \nbest_eval_score {best_eval_score}')
            print(f"Epoch {epoch + 1} Evaluation Results: {eval_results}")

            if current_score > best_eval_score:
                best_eval_score = current_score
                
                # Save the model and tokenizer
                model.save_pretrained(model_save_path)
                tokenizer.save_pretrained(model_save_path)
                peft_config.save_pretrained(model_save_path)
                print(f"Model saved at {model_save_path} with F1 Score: {best_eval_score}")
    except:
        raise
###########################################################################################################################
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def prepare_contexts(context_list, citation=True):
    context_text = []
    if citation:
        for i in range(len(context_list)) :
            doc = "[" + str((i + 1)) + "] " + context_list[i]
            context_text.append(doc)
    else:
        for i in range(len(context_list)) :
            doc = context_list[i]
            context_text.append(doc)
        
    return "\n".join(context_text)

if __name__ == "__main__":
    main()

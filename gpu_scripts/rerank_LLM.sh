#!/bin/bash

#SBATCH --partition=hard

#SBATCH --job-name=rerank_LLM 

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1  

#SBATCH --time=5-00:00:00

#SBATCH --output=rerank_LLM.out

#SBATCH --error=rerank_LLM.err

#SBATCH --ntasks-per-node=1 

python scripts/rerank_LLM.py --data_set comp24 --input_file doc_vanilla_2024_10.tsv --corpus segmented



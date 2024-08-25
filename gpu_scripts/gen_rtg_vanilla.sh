#!/bin/bash

#SBATCH --partition=hard

#SBATCH --job-name=rtg_v3_p13_llama

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1  

#SBATCH --time=5-00:00:00

#SBATCH --output=rtg_v3_p13_llama.out

#SBATCH --error=rtg_v3_p13_llama.err

#SBATCH --ntasks-per-node=1 

#SBATCH --exclude=led,lizzy,thin,zeppelin

python scripts/gen_answ.py --model_name llama2 --data_set comp24 --architcture RTG-vanilla --output_file answer_v3_p13_llama.jsonl --input_file doc_vanilla_2024_100.tsv


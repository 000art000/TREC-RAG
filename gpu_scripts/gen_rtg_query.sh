#!/bin/bash

#SBATCH --partition=hard

#SBATCH --job-name=rtg_q3_p13_llama

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1  

#SBATCH --time=5-00:00:00

#SBATCH --output=rtg_q3_p13_llama.out

#SBATCH --error=rtg_q3_p13_llama.err

#SBATCH --ntasks-per-node=1 

#SBATCH --exclude=led,lizzy,thin,zeppelin

python scripts/gen_answ.py --model_name llama2 --data_set comp24 --architcture RTG-query-gen --output_file answer_q3_p13_llama.jsonl --input_file doc_rtg_gen_2024_100.tsv


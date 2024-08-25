#!/bin/bash

#SBATCH --partition=hard,electronic

#SBATCH --job-name=citations_eval_q2_p13_llama

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1  

#SBATCH --time=5-00:00:00

#SBATCH --output=citations_eval_q2_p13_llama.out

#SBATCH --error=citations_eval_q2_p13_llama.err

#SBATCH --ntasks-per-node=1 

python scripts/citations_eval.py --output_file answer_q2_p13_llama.jsonl --architcture RTG-query-gen
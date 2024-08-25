#!/bin/bash

#SBATCH --partition=electronic,hard

#SBATCH --job-name=gen_vanilla 

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1  

#SBATCH --time=5-00:00:00

#SBATCH --output=gen_vanilla.out

#SBATCH --error=gen_vanilla.err

#SBATCH --ntasks-per-node=1 

python scripts/generate_answer.py --data_set MS_MARCO_Dev2 --results_file G_result.csv --output_file G_output.jsonl


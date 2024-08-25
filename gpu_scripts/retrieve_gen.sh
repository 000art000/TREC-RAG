#!/bin/bash

#SBATCH --partition=hard,electronic

#SBATCH --job-name=ret_gen 

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1  

#SBATCH --time=5-00:00:00

#SBATCH --output=ret_gen.out

#SBATCH --error=ret_gen.err

#SBATCH --ntasks-per-node=1 

python scripts/retrieve_with_generated_queries.py --output_file doc_rtg_gen_2024_100.tsv
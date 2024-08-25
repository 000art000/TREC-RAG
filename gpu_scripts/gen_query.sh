#!/bin/bash

#SBATCH --partition=electronic

#SBATCH --job-name=gen_query 

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1  

#SBATCH --time=5-00:00:00

#SBATCH --output=gen_query.out

#SBATCH --error=gen_query.err

#SBATCH --ntasks-per-node=1 

python scripts/generate_queries.py --data_set comp24


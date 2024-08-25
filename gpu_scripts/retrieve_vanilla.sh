#!/bin/bash

#SBATCH --partition=hard,electronic,jazzy,funky

#SBATCH --job-name=ret_vanilla 

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1  

#SBATCH --time=5-00:00:00

#SBATCH --output=ret_vanilla.out

#SBATCH --error=ret_vanilla.err

#SBATCH --ntasks-per-node=1 

#SBATCH --exclude=led,lizzy,thin,zeppelin

python scripts/retrieve.py --data_set comp24 --output_file doc_vanilla_2024_10.tsv
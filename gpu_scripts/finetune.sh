#!/bin/bash

#SBATCH --partition=hard

#SBATCH --job-name=finetune_3

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1  

#SBATCH --time=5-00:00:00

#SBATCH --output=finetune_3.out

#SBATCH --error=finetune_3.err

#SBATCH --ntasks-per-node=1

#SBATCH --exclude=led,lizzy,thin,zeppelin

python scripts/finetune_2.py --repetoire model_t5_large


#!/bin/bash

#SBATCH -p lyceum

#SBATCH --time=60:00:00          # walltime
#SBATCH --gres=gpu:1

python inference.py
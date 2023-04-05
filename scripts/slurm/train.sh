#!/bin/bash
#SBATCH -J train                        # Job name
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 16                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=64000                           # server memory requested (per node)
#SBATCH -t 2:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=default_partition       # Request partition
#SBATCH --gres=gpu:1080ti:4                  # Type/number of GPUs needed
python3 /home/tjp93/foodrecognition/model/train.py
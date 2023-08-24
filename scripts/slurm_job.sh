#!/usr/bin/env bash

#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/job-%j.out
#SBATCH --mem-per-gpu=12G

apptainer run --nv --writable-tmpfs --env-file .env images/image.sif /usr/bin/python temporal_fusion_transformer/main.py \
  --batch_size=512 --mixed_precision=true --jit_module=true --task=model --experiment=electricity

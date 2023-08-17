#!/bin/bash

#SBATCH --job-name=build_image
#SBATCH --partition=cpu-2h
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

apptainer build images/base_image.sif images/base_image.def
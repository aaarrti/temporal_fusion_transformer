#!/bin/bash

#SBATCH --job-name=build_image
#SBATCH --partition=cpu-2h
#SBATCH --output=logs/job-%j.out

set -ex
rm -f images/image.sif
apptainer build --fakeroot images/image.sif images/image.def
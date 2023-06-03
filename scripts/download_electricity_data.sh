#!/usr/bin/env zsh

set -ex
mkdir -p data/electricity
cd data/electricity
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip
unzip LD2011_2014.txt.zip

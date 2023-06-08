#!/usr/bin/env zsh

set -ex
mkdir -p data/electricity
cd data/electricity
wget https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
unzip electricityloaddiagrams20112014.zip
rm -r __MACOSX

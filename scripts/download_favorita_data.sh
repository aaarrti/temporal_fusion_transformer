#!/usr/bin/env zsh

set -ex

mkdir -p raw_data/favorita
cd raw_data/favorita
kaggle competitions download -c favorita-grocery-sales-forecasting
unzip favorita-grocery-sales-forecasting.zip

7z e holidays_events.csv.7z
7z e items.csv.7z
7z e oil.csv.7z
7z e sample_submission.csv.7z
7z e stores.csv.7z
7z e test.csv.7z
7z e train.csv.7z
7z e transactions.csv.7z

rm *.7z
rm favorita-grocery-sales-forecasting.zip

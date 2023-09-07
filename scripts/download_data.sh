#!/usr/bin/env zsh

set -ex
TARGET=$1

if [ "$TARGET" = "electricity" ]; then
  mkdir -p data/electricity
  cd data/electricity
  kaggle datasets download -d sanketbodake1998/demand-energy-forecasting1
  unzip demand-energy-forecasting1.zip
  rm demand-energy-forecasting1.zip
elif [ "$TARGET" = "favorita" ]; then
  mkdir -p data/favorita
  cd data/favorita
  kaggle competitions download -c favorita-grocery-sales-forecasting
  unzip favorita-grocery-sales-forecasting.zip
  rm favorita-grocery-sales-forecasting.zip
  for i in holidays_events items oil stores train transactions
  do
    7z e "$i".csv.7z
    rm "$i".csv.7z
  done
  rm sample_submission.7z.csv
  rm test.7z.csv
fi




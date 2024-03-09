#!/usr/bin/env zsh

set -ex
TARGET=$1
mkdir -p data/"$TARGET"
cd data/"$TARGET"

if [ "$TARGET" = "electricity" ]; then
  kaggle datasets download -d sanketbodake1998/demand-energy-forecasting1
  unzip demand-energy-forecasting1.zip
  rm demand-energy-forecasting1.zip
elif [ "$TARGET" = "favorita" ]; then
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
elif [ "$TARGET" = "air_passengers" ]; then
  wget https://raw.githubusercontent.com/unit8co/darts/master/datasets/AirPassengers.csv
elif [ "$TARGET" = "ice_cream_heater" ]; then
  wget https://raw.githubusercontent.com/unit8co/darts/master/datasets/ice_cream_heater.csv

fi




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
fi




## TensorFlow2 reimplementation of Temporal Fusion Transformer (Work In Progress 🚧)

### It supports:

- TF2 eager execution ✅
- `tf.data` APIs ✅
- XLA compiler 🚧
- TPU ✅
- Pre-processing with `polars` and `pyarrow` 🚧
- mixed_precision ✅

The goal is to repeat the 4 experiments done in the original publication.

- Electricity ✅
- Favorite 🚧
- Traffic 🚧
- Volatility 🚧

The final goal however, is to have a production grade implementation, with straight-forward API
(which is not the case for original implementation) for extending it for custom datasets,
as well as to utilize performance improving features of TF2 (which is also not the case for original implementation).

## References

- The implementation as well as experimental setting closely follow the ones described by Bryan Lim, Sercan O. Arik,
  Nicolas Loeff, Tomas Pfister
  in their
  publication [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
- The codebase is mostly adapted
  from [google-research/google-research/tft](https://github.com/google-research/google-research/tree/master/tft)
  repository.
  If you need a local copy of it, run: `./scripts/clone_original_implementation.sh`

### Local development how-to

#### Updating dataset

```shell
./scripts/download_<expriment name>_data
python scripts/export_tensorflow_dataset.py --dataset=<expriment name
dvc add data
dvc commit
dvc push
```

### Upload dataset to GCS bucket to be used by colab TPUs.

```shell
gcloud auth login
terraform init
terraform plan
terraform apply
gsutil cp -r data gs://tf2_tft_v2/
```

### TODO:
- create v2 models without RNN-s and LSTM-s (like full blown transformer).
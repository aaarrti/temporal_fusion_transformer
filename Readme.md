## Flax reimplementation of Temporal Fusion Transformer


#### Experiments
Two experiments from the original publication are duplicated in this repository.
- Electricity ✅
- Favorita 🚧

Additionally, we apply this model to [Hamburg Air Quality](https://repos.hcu-hamburg.de/handle/hcu/893) dataset,
and compare forecasts with [BigQuery ML](https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create-time-series).

#### Datasets
You don't need `tensorflow` installation to run the model, however, 
we use `tensorflow.data` for data loading, preprocessing, etc., as there is no better option up-to-date.

#### Using with your data  🚧
TODO

Joins required for `Favorita` dataset barely fit into memory, so we offload them [Big Query](https://cloud.google.com/bigquery/docs).
In theory, you could also use [AWS Athena](https://aws.amazon.com/athena/) for you dataset.

#### Training
The training scripts in this repository are designed to run on [Cloud TPU](https://cloud.google.com/tpu).

### Inference 🚧
OpenXLA IREE?

## References 

- The implementation as well as experimental setting closely follow the ones described by Bryan Lim, Sercan O. Arik,
  Nicolas Loeff, Tomas Pfister
  in their
  publication [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
- The codebase is mostly adapted
  from [google-research/google-research/tft](https://github.com/google-research/google-research/tree/master/tft)
  repository.
  If you need a local copy of it, run: `./scripts/clone_original_implementation.sh`

## Differences with original implementation
- Supports multiple transformer blocks stacked
- No attention output, as attention-based explanations do not fulfill sanity-checks (TODO: `quantus link`)
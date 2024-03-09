# JAX reimplementation of Temporal Fusion Transformer

---

## ❌

Unfortunately, despite my preference for `jax` + `flax` I had to abandon this project
in favour of (successful) existing and maintained `torch` implementations (*sigh).

- [x] [gluonts](https://github.com/awslabs/gluonts)
  ➡️ [TemporalFusionTransformerEstimator](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/model/tft/estimator.py)
- [x] [darts](https://github.com/unit8co/darts)
  ➡️ [TFT Examples](https://github.com/unit8co/darts/blob/master/examples/13-TFT-examples.ipynb)

---

### References

- The implementation as well as experimental setting closely follow the ones described by Bryan Lim, Sercan O. Arik,
  Nicolas Loeff, Tomas Pfister
  in their
  publication [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
- The codebase is mostly adapted
  from [google-research/google-research/tft](https://github.com/google-research/google-research/tree/master/tft)
  repository.
  If you need a local copy of it, run: `./scripts/clone_original_implementation.sh`

---

### Build `.sif` image

```shell
srun --partition=cpu-2h --ntasks-per-node=2 --pty bash
apptainer build --fakeroot images/image.sif images/image.def
```
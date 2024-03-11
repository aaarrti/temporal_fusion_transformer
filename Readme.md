## JAX reimplementation of Temporal Fusion Transformer

---

### ❌Actually, I hated darts so much, that I picked it up back again.

### 🚧 ... 🛠️

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
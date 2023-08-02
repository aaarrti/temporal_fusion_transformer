#### Upload data to GCS
```shell
gsutil cp -r data gs://tft_experiments/
```
#### Create SquashFS image
```shell
mksquashfs data data.sqfs -all-root -action 'chmod(o+rX)@!perm(o+rX)'
```

#### Scripts must be run from project root, e.g.,
```shell
./scripts/download_favorita_data.sh 
```

---
### TODO:
- lifted jit
- pad-shard un-pad
- train/eval model, avoid recompilation
- TPU training
- mixed precision / pass down `dtype`?

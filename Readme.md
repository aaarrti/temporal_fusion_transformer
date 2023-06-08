# Work In Progress

#### This is TF2 re-implemetation of (TODO insert link) Temporal Fusion Transformer

### TODO references

mksquashfs data data.sqfs -all-root -action 'chmod(o+rX)@!perm(o+rX)

Run `./scripts/clone_original_implementation`, if you need to compare it with this repository.
Run `./scripts/download_<expriment name>_data`, to download raw data from experiment.

### Uploading dataset

```shell
gcloud auth login
cd gcs
terraform init
terraform plan
terraform apply
cd ..
gsutil cp -r data gs://tf2_tft/
```


```
[{{node QuantileLoss/Reshape}}]]
[[TPUReplicate/_compile/_11505735876197561850/_2]]
[[tpu_compile_succeeded_assert/_2597365915436670702/_3/_71]]
(4) INVALID_ARGUMENT: {{function_node __inference_train_function_80028}} Input 1 to node `QuantileLoss/Reshape` with op Reshape must be a compile-time constant.

XLA compilation requires that operator arguments that represent shapes or dimensions be evaluated to concrete values at compile  ... [truncated]
```

```
lstm do not support bfloats
```

```
access for service-495559152420@cloud-tpu.iam.gserviceaccount.com
```
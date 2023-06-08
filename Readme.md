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
#### Build base `.sif` image
```shell
srun --partition=cpu-2h --ntasks-per-node=2 --pty bash
apptainer build --fakeroot images/base_image.sif images/base_image.def
```

### Build `.sif` image

```shell
srun --partition=cpu-2h --ntasks-per-node=2 --pty bash
apptainer build --fakeroot images/image.sif images/image.def
```
#### Allowing Colab TPU to access GCS bucket.

1. Go to [Colab](https://colab.research.google.com/)
2. Kick off a runtime with TPU
3. Authorize yourself

```shell
!gcloud auth login
!gcloud config set project tm-mapping
```

```python
from google.colab import auth

auth.authenticate_user()
```

4. Initialize TPU

```python
import tensorflow as tf
import os

tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)
```

5. Try load something from `GCS` bucket, e.g.,

```python
tf.keras.models.load_model("gs://tm_mapping/models/labse_tiny/model/")
```

6. At this point you will see something like:

```shell
{
  "code": 403,
  "message": "service-495559152420@cloud-tpu.iam.gserviceaccount.com does not have storage.objects.list access to the Google Cloud Storage bucket. Permission 'storage.objects.list' denied on resource (or it may not exist).",
}
```

7. Here you have your service account id is:

```terraform
variable "tpu_sa" {
  default     = "serviceAccount:service-495559152420@cloud-tpu.iam.gserviceaccount.com"
  description = "Service account used by Colab TPU"
}
```
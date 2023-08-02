variable "project_id" {
  description = "id of the GCP project"
  default     = "titanium-atlas-389220"
}

variable "location" {
  description = "Cloud region to use"
  default     = "europe-west4"
}

variable "bucket-name" {
  default     = "tft_datasets"
  description = "Name of the GCS bucket"
}

variable "tpu_sa" {
  default     = "serviceAccount:service-891405943222@cloud-tpu.iam.gserviceaccount.com"
  description = "Service account used by Colab TPU"
}


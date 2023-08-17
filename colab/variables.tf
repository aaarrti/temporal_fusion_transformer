// Just for convenience store constants here.
variable "project-id" {
  default = "tm-mapping"
}

variable "region" {
  default = "europe-west4"
}

variable "bucket-name" {
  default     = "tm_mapping"
  description = "Name of the GCS bucket"
}

variable "tpu_sa" {
  default     = "serviceAccount:service-495559152420@cloud-tpu.iam.gserviceaccount.com"
  description = "Service account used by Colab TPU"
}
module "bucket" {
  source     = "terraform-google-modules/cloud-storage/google//modules/simple_bucket"
  version    = "4.0.0"
  name       = var.bucket-name
  project_id = var.project_id
  location   = var.location
}
module "bigquery" {
  source       = "terraform-google-modules/bigquery/google"
  version      = "6.1.1"
  dataset_id   = "favorita"
  dataset_name = "favorita"
  project_id   = var.project_id
  location     = var.location
}

// ----
resource "google_storage_bucket_iam_member" "TPU_ROLE_1" {
  bucket     = var.bucket-name
  role       = "roles/storage.objectViewer"
  member     = var.tpu_sa
  depends_on = [module.bucket]
}

resource "google_storage_bucket_iam_member" "TPU_ROLE_2" {
  bucket     = var.bucket-name
  role       = "roles/storage.legacyBucketReader"
  member     = var.tpu_sa
  depends_on = [module.bucket]
}

resource "google_storage_bucket_iam_member" "TPU_ROLE_3" {
  bucket     = var.bucket-name
  role       = "roles/storage.objectCreator"
  member     = var.tpu_sa
  depends_on = [module.bucket]
}

resource "google_storage_bucket_iam_member" "TPU_ROLE_4" {
  bucket     = var.bucket-name
  role       = "roles/storage.legacyBucketWriter"
  member     = var.tpu_sa
  depends_on = [module.bucket]
}
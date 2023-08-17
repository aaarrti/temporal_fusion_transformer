resource "google_storage_bucket" "data_bucket" {
  name                        = var.bucket-name
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  force_destroy               = true
}
// ----
resource "google_storage_bucket_iam_member" "TPU_ROLE_1" {
  bucket     = var.bucket-name
  role       = "roles/storage.objectViewer"
  member     = var.tpu_sa
  depends_on = [google_storage_bucket.data_bucket]
}

resource "google_storage_bucket_iam_member" "TPU_ROLE_2" {
  bucket     = var.bucket-name
  role       = "roles/storage.legacyBucketReader"
  member     = var.tpu_sa
  depends_on = [google_storage_bucket.data_bucket]
}

resource "google_storage_bucket_iam_member" "TPU_ROLE_3" {
  bucket     = var.bucket-name
  role       = "roles/storage.objectCreator"
  member     = var.tpu_sa
  depends_on = [google_storage_bucket.data_bucket]
}

resource "google_storage_bucket_iam_member" "TPU_ROLE_4" {
  bucket     = var.bucket-name
  role       = "roles/storage.legacyBucketWriter"
  member     = var.tpu_sa
  depends_on = [google_storage_bucket.data_bucket]
}
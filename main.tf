provider "google" {
  project = "titanium-atlas-389220"
  region  = "europe-west4"
}

module "bucket" {
  source  = "terraform-google-modules/cloud-storage/google//modules/simple_bucket"
  version = "~> 4.0"

  name        = "tf2_tft_v2"
  project_id  = "titanium-atlas-389220"
  location    = "europe-west4"
  iam_members = [
    {
      role   = "roles/storage.objectAdmin"
      member = "user:artem.sereda.tub@gmail.com"
    },
    {
      role   = "roles/storage.legacyBucketReader"
      member = "serviceAccount:service-495559152420@cloud-tpu.iam.gserviceaccount.com"
    },
    {
      role   = "roles/storage.objectViewer"
      member = "serviceAccount:service-495559152420@cloud-tpu.iam.gserviceaccount.com"
    },
    {
      role   = "roles/storage.objectCreator"
      member = "serviceAccount:service-495559152420@cloud-tpu.iam.gserviceaccount.com"
    },
  ]
}
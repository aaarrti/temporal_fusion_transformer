provider "google" {
  project = "titanium-atlas-389220"
  region  = "europe-west4"
}

module "bucket" {
  source  = "terraform-google-modules/cloud-storage/google//modules/simple_bucket"
  version = "~> 4.0"

  name        = "tf2_tft"
  project_id  = "titanium-atlas-389220"
  location    = "europe-west4"
  iam_members = [
    {
      role   = "roles/storage.objectAdmin"
      member = "user:artem.sereda.tub@gmail.com"
    },
    # If I can auth in colab, it is fine.
    #{
    #  role   = "roles/storage.objectViewer"
    #  member = "allUsers"
    #},
  ]
}
terraform {
  backend "s3" {
    bucket         = "uav-link-quality-routing-support-tfstate-staging"
    key            = "staging/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "uav-link-quality-routing-support-tfstate-staging-lock"
    profile        = "default"
  }
}

output "s3_bucket_name" {
  value = aws_s3_bucket.mlops_artifacts.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app_repo.repository_url
}

output "rds_endpoint" {
  value = aws_db_instance.mlflow_db.endpoint
}

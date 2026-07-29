output "train_repository_url" {
  description = "Repository URL for the training image repository"
  value       = aws_ecr_repository.train.repository_url
}

output "serve_repository_url" {
  description = "Repository URL for the serving image repository"
  value       = aws_ecr_repository.serve.repository_url
}

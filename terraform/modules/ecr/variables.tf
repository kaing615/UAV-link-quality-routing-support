variable "train_repo_name" {
  description = "Name of the training ECR repository"
  type        = string
  default     = "train"
}

variable "serve_repo_name" {
  description = "Name of the serving ECR repository"
  type        = string
  default     = "serve"
}

variable "tags" {
  description = "Tags to apply to ECR repositories"
  type        = map(string)
  default     = {}
}

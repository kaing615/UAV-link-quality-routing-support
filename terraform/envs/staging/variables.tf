variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "uav-gnn"
}

variable "environment" {
  type    = string
  default = "staging"
}

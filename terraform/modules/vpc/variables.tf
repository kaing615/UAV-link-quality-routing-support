variable "project_name" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Environment (staging/production)"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

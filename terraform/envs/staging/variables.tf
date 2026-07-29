variable "aws_region" {
  description = "AWS region for the staging environment"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name prefix for resources"
  type        = string
  default     = "uav-link-quality-routing-support"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones for subnets"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.11.0/24"]
}

variable "enable_nat_gateway" {
  description = "Whether to create NAT gateways"
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "Whether to create a single shared NAT gateway"
  type        = bool
  default     = true
}

variable "cpu_node_instance_type" {
  description = "Instance type for the CPU node group"
  type        = string
  default     = "t3.medium"
}

variable "gpu_node_instance_type" {
  description = "Instance type for the GPU node group"
  type        = string
  default     = "g4dn.xlarge"
}

variable "mlflow_db_username" {
  description = "Master username for the MLflow PostgreSQL instance"
  type        = string
  default     = "mlflowadmin"
}

variable "mlflow_db_password" {
  description = "Master password for the MLflow PostgreSQL instance"
  type        = string
  sensitive   = true
  default     = "ChangeMe123!"
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Environment = "staging"
    ManagedBy   = "terraform"
  }
}

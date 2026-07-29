variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for the EKS cluster and node groups"
  type        = list(string)
}

variable "cpu_node_instance_type" {
  description = "Instance type for CPU node group"
  type        = string
  default     = "t3.medium"
}

variable "cpu_node_disk_size" {
  description = "Disk size for CPU node group"
  type        = number
  default     = 50
}

variable "cpu_node_desired_size" {
  type    = number
  default = 2
}

variable "cpu_node_max_size" {
  type    = number
  default = 3
}

variable "cpu_node_min_size" {
  type    = number
  default = 1
}

variable "gpu_node_instance_type" {
  description = "Instance type for GPU node group"
  type        = string
  default     = "g4dn.xlarge"
}

variable "gpu_node_disk_size" {
  description = "Disk size for GPU node group"
  type        = number
  default     = 80
}

variable "gpu_node_desired_size" {
  type    = number
  default = 1
}

variable "gpu_node_max_size" {
  type    = number
  default = 2
}

variable "gpu_node_min_size" {
  type    = number
  default = 0
}

variable "tags" {
  description = "Tags to apply to EKS resources"
  type        = map(string)
  default     = {}
}

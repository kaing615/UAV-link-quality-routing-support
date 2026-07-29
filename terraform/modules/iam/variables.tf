variable "name" {
  description = "Name of the IAM role"
  type        = string
}

variable "assume_role_policy" {
  description = "JSON policy document for the role trust relationship"
  type        = string
}

variable "managed_policy_arns" {
  description = "List of AWS managed policy ARNs to attach"
  type        = list(string)
  default     = []
}

variable "inline_policies" {
  description = "Inline policy documents to attach to the role"
  type = list(object({
    name   = string
    policy = string
  }))
  default = []
}

variable "path" {
  description = "Path for the IAM role"
  type        = string
  default     = "/"
}

variable "description" {
  description = "Description of the IAM role"
  type        = string
  default     = null
}

variable "tags" {
  description = "Tags to apply to the IAM role"
  type        = map(string)
  default     = {}
}

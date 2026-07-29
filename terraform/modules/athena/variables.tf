variable "database_name" {
  description = "Name of the Glue database"
  type        = string
}

variable "table_name" {
  description = "Name of the Glue table"
  type        = string
}

variable "bucket_name" {
  description = "S3 bucket location for the Glue table"
  type        = string
}

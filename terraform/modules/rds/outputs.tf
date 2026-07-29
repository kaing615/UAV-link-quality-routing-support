output "endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.this.address
}

output "port" {
  description = "RDS instance port"
  value       = aws_db_instance.this.port
}

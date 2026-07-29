resource "aws_db_subnet_group" "this" {
  name       = var.identifier
  subnet_ids = var.subnet_ids

  tags = merge({ Name = var.identifier }, var.tags)
}

resource "aws_db_instance" "this" {
  identifier              = var.identifier
  allocated_storage       = var.allocated_storage
  storage_type            = "gp3"
  engine                  = "postgres"
  engine_version          = var.engine_version
  instance_class          = var.instance_class
  db_name                 = var.db_name
  username                = var.username
  password                = var.password
  db_subnet_group_name    = aws_db_subnet_group.this.name
  vpc_security_group_ids  = var.vpc_security_group_ids
  publicly_accessible     = false
  multi_az                = true
  backup_retention_period = 14
  backup_window           = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  skip_final_snapshot     = false
  final_snapshot_identifier = "${var.identifier}-final"
  storage_encrypted       = true

  tags = merge({ Name = var.identifier }, var.tags)
}

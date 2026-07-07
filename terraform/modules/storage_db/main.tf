# 1. Amazon S3 for DVC and MLflow Artifacts
resource "aws_s3_bucket" "mlops_artifacts" {
  bucket = "${var.project_name}-${var.environment}-mlops-artifacts"
}

resource "aws_s3_bucket_versioning" "mlops_artifacts_versioning" {
  bucket = aws_s3_bucket.mlops_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# 2. Amazon ECR for Docker images
resource "aws_ecr_repository" "app_repo" {
  name                 = "${var.project_name}-${var.environment}-repo"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# 3. Amazon RDS for MLflow Tracking Server
resource "aws_db_subnet_group" "db_subnet" {
  name       = "${var.project_name}-${var.environment}-db-subnet"
  subnet_ids = var.subnet_ids
}

resource "aws_security_group" "rds_sg" {
  name        = "${var.project_name}-${var.environment}-rds-sg"
  description = "Allow Postgres traffic from EKS"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"] # Allow from within VPC
  }
}

resource "aws_db_instance" "mlflow_db" {
  identifier           = "${var.project_name}-${var.environment}-mlflow-db"
  allocated_storage    = 20
  storage_type         = "gp2"
  engine               = "postgres"
  engine_version       = "15.4"
  instance_class       = "db.t3.micro"
  db_name              = "mlflowdb"
  username             = "mlflow_user"
  password             = "SuperSecretPassword123!" # In real life, use AWS Secrets Manager
  parameter_group_name = "default.postgres15"
  skip_final_snapshot  = true
  
  db_subnet_group_name   = aws_db_subnet_group.db_subnet.name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
}

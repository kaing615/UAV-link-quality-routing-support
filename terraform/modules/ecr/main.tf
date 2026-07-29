resource "aws_ecr_repository" "train" {
  name                 = var.train_repo_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = merge({ Name = var.train_repo_name }, var.tags)
}

resource "aws_ecr_repository" "serve" {
  name                 = var.serve_repo_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = merge({ Name = var.serve_repo_name }, var.tags)
}

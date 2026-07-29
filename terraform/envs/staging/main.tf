terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.14"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.31"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    token                  = data.aws_eks_cluster_auth.this.token
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_cluster_auth.this.token
}

provider "tls" {}

module "vpc" {
  source = "../../modules/vpc"

  name                 = var.project_name
  cidr_block           = var.vpc_cidr
  availability_zones   = var.availability_zones
  public_subnet_cidrs  = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs
  enable_nat_gateway   = var.enable_nat_gateway
  single_nat_gateway   = var.single_nat_gateway
  tags                 = var.tags
}

module "eks" {
  source = "../../modules/eks"

  cluster_name           = "${var.project_name}-staging"
  subnet_ids             = module.vpc.private_subnet_ids
  cpu_node_instance_type = var.cpu_node_instance_type
  gpu_node_instance_type = var.gpu_node_instance_type
  tags                   = var.tags
}

resource "aws_security_group" "rds" {
  name        = "${var.project_name}-rds-sg"
  description = "Allow PostgreSQL traffic from the VPC"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "tls_certificate" "eks" {
  url = module.eks.cluster_oidc_issuer_url
}

data "aws_eks_cluster_auth" "this" {
  name = module.eks.cluster_name
}

resource "aws_iam_openid_connect_provider" "eks" {
  url             = module.eks.cluster_oidc_issuer_url
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks.certificates[0].sha1_fingerprint]
}

module "ecr" {
  source = "../../modules/ecr"

  train_repo_name = "${var.project_name}/train"
  serve_repo_name = "${var.project_name}/serve"
  tags            = var.tags
}

module "dvc_remote" {
  source = "../../modules/s3"

  bucket_name = "${var.project_name}-dvc-staging"
  tags        = var.tags
}

module "prediction_store" {
  source = "../../modules/s3"

  bucket_name = "${var.project_name}-predictions-staging"
  tags        = var.tags
}

module "athena" {
  source = "../../modules/athena"

  database_name = "${replace(var.project_name, "-", "_")}_predictions"
  table_name    = "predictions"
  bucket_name   = module.prediction_store.bucket_id
}

module "rds" {
  source = "../../modules/rds"

  identifier             = "${var.project_name}-mlflow"
  db_name                = "mlflow"
  username               = var.mlflow_db_username
  password               = var.mlflow_db_password
  subnet_ids             = module.vpc.private_subnet_ids
  vpc_security_group_ids = [aws_security_group.rds.id]
  tags                   = var.tags
}

module "irsa_alb_controller" {
  source = "../../modules/iam-irsa"

  role_name                  = "${var.project_name}-alb-controller"
  policy_arn                 = "arn:aws:iam::aws:policy/AWSLoadBalancerControllerIAMPolicy"
  oidc_provider_arn          = aws_iam_openid_connect_provider.eks.arn
  oidc_provider_url          = aws_iam_openid_connect_provider.eks.url
  kubernetes_namespace       = "kube-system"
  kubernetes_service_account = "aws-load-balancer-controller"
}

module "irsa_external_dns" {
  source = "../../modules/iam-irsa"

  role_name                  = "${var.project_name}-external-dns"
  policy_arn                 = "arn:aws:iam::aws:policy/AmazonRoute53FullAccess"
  oidc_provider_arn          = aws_iam_openid_connect_provider.eks.arn
  oidc_provider_url          = aws_iam_openid_connect_provider.eks.url
  kubernetes_namespace       = "kube-system"
  kubernetes_service_account = "external-dns"
}

module "irsa_cert_manager" {
  source = "../../modules/iam-irsa"

  role_name                  = "${var.project_name}-cert-manager"
  policy_arn                 = "arn:aws:iam::aws:policy/AmazonRoute53FullAccess"
  oidc_provider_arn          = aws_iam_openid_connect_provider.eks.arn
  oidc_provider_url          = aws_iam_openid_connect_provider.eks.url
  kubernetes_namespace       = "cert-manager"
  kubernetes_service_account = "cert-manager"
}

resource "helm_release" "argocd" {
  name             = "argocd"
  namespace        = "argocd"
  create_namespace = true
  repository       = "https://argoproj.github.io/argo-helm"
  chart            = "argo-cd"
  version          = "7.5.2"
  timeout          = 1800

  values = [yamlencode({
    server = {
      service = {
        type = "LoadBalancer"
      }
    }
    configs = {
      params = {
        "server.insecure" = true
      }
    }
  })]

  depends_on = [module.eks]
}

resource "kubernetes_manifest" "argocd_root_app" {
  manifest = yamldecode(file("../../../deploy/argocd/root-app.yaml"))

  depends_on = [helm_release.argocd]
}

module "irsa_app_s3" {
  source = "../../modules/iam-irsa"

  role_name                  = "${var.project_name}-app-s3"
  policy_arn                 = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
  oidc_provider_arn          = aws_iam_openid_connect_provider.eks.arn
  oidc_provider_url          = aws_iam_openid_connect_provider.eks.url
  kubernetes_namespace       = "default"
  kubernetes_service_account = "app-s3"
}

module "irsa_prediction_writer" {
  source = "../../modules/iam-irsa"

  role_name                  = "${var.project_name}-prediction-writer"
  policy_arn                 = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
  oidc_provider_arn          = aws_iam_openid_connect_provider.eks.arn
  oidc_provider_url          = aws_iam_openid_connect_provider.eks.url
  kubernetes_namespace       = "default"
  kubernetes_service_account = "prediction-writer"
}

resource "aws_budgets_budget" "monthly_cost" {
  name              = "${var.project_name}-monthly-budget"
  budget_type       = "COST"
  limit_amount      = "100"
  limit_unit        = "USD"
  time_unit         = "MONTHLY"
  time_period_start = "2026-07-01_00:00"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["ops@example.com"]
  }
}

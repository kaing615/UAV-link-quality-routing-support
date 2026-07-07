module "vpc" {
  source       = "../../modules/vpc"
  project_name = var.project_name
  environment  = var.environment
  vpc_cidr     = "10.0.0.0/16"
}

module "eks" {
  source       = "../../modules/eks"
  project_name = var.project_name
  environment  = var.environment
  vpc_id       = module.vpc.vpc_id
  subnet_ids   = module.vpc.private_subnets
  
  depends_on = [module.vpc]
}

module "storage_db" {
  source       = "../../modules/storage_db"
  project_name = var.project_name
  environment  = var.environment
  vpc_id       = module.vpc.vpc_id
  subnet_ids   = module.vpc.private_subnets
  
  depends_on = [module.vpc]
}

output "eks_cluster_name" {
  value = module.eks.cluster_name
}

output "s3_bucket" {
  value = module.storage_db.s3_bucket_name
}

output "rds_endpoint" {
  value = module.storage_db.rds_endpoint
}

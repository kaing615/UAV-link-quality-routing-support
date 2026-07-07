module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "${var.project_name}-${var.environment}-cluster"
  cluster_version = "1.30"

  cluster_endpoint_public_access  = true
  
  vpc_id                   = var.vpc_id
  subnet_ids               = var.subnet_ids

  eks_managed_node_groups = {
    # CPU Node group for serving and platform tools
    cpu_nodes = {
      min_size     = 1
      max_size     = 3
      desired_size = 2

      instance_types = ["t3.medium"]
      capacity_type  = "SPOT"
    }

    # GPU Node group for GNN training/inference
    gpu_nodes = {
      min_size     = 0
      max_size     = 1
      desired_size = 0

      instance_types = ["g4dn.xlarge"]
      capacity_type  = "SPOT"
      
      ami_type = "AL2_x86_64_GPU"

      taints = {
        dedicated = {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }

  tags = {
    Environment = var.environment
    Project     = var.project_name
    Terraform   = "true"
  }
}

# Providers and remote state backend.
#
# The S3 backend is partially configured - bucket, key, region and
# dynamodb_table are supplied at `terraform init` via -backend-config
# flags (see .github/workflows/ci-cd.yml and infra/bootstrap/).

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.0" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }

  backend "s3" {}
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

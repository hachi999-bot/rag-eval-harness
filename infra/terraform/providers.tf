# AWS ECS Fargate deployment example / reference architecture for the
# rag-eval-harness REST API. See README.md — this configuration illustrates
# a possible production layout and is not applied as part of this
# repository's normal workflow; running `terraform apply` against a real
# AWS account is a separate, explicit action left to whoever uses it.

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

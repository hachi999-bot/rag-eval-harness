variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "ap-northeast-1"
}

variable "project_name" {
  description = "Name prefix used for all resources created by this configuration."
  type        = string
  default     = "rag-eval-harness"
}

variable "vpc_id" {
  description = "ID of an existing VPC to deploy the ECS service into. This configuration does not create a VPC."
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs (in the existing VPC given by var.vpc_id) for the Fargate service's ENIs."
  type        = list(string)
}

variable "security_group_ids" {
  description = "Existing security group IDs to attach to the Fargate service's ENIs."
  type        = list(string)
}

variable "container_port" {
  description = "Port the API listens on inside the container (must match the Dockerfile's EXPOSE / uvicorn --port)."
  type        = number
  default     = 8000
}

variable "task_cpu" {
  description = "Fargate task-level CPU units (see AWS Fargate CPU/memory combinations)."
  type        = string
  default     = "256"
}

variable "task_memory" {
  description = "Fargate task-level memory in MiB."
  type        = string
  default     = "512"
}

variable "desired_count" {
  description = "Number of Fargate tasks the ECS service keeps running."
  type        = number
  default     = 1
}

variable "assign_public_ip" {
  description = "Whether the Fargate task's ENI gets a public IP. Set true only if the given subnets are public and there is no NAT gateway for pulling the image / reaching CloudWatch Logs."
  type        = bool
  default     = false
}

variable "image_tag" {
  description = "Tag of the image in the ECR repository this configuration creates, to run in the task definition."
  type        = string
  default     = "latest"
}

variable "log_retention_days" {
  description = "CloudWatch Logs retention period, in days, for the task's log group."
  type        = number
  default     = 14
}

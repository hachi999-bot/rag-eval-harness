output "ecr_repository_url" {
  description = "URL of the ECR repository the container image should be pushed to before the ECS service can start."
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster."
  value       = aws_ecs_cluster.app.name
}

output "ecs_service_name" {
  description = "Name of the ECS service."
  value       = aws_ecs_service.app.name
}

output "cloudwatch_log_group" {
  description = "CloudWatch Logs group receiving the container's stdout/stderr."
  value       = aws_cloudwatch_log_group.app.name
}

output "task_execution_role_arn" {
  description = "IAM role ARN ECS assumes to pull the image from ECR and write logs."
  value       = aws_iam_role.execution.arn
}

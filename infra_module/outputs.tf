output "resource_benchmark_public_ip" {
  description = "Public IP of the instance"
  value       = aws_instance.ml_benchmark.public_ip
}
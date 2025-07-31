output "ml_benchmark_public_ips" {
  description = "Public IPs of all ml benchmarking instances"
  value       = { for k, m in module.infra_benchmarks : k => m.resource_benchmark_public_ip }
}

output "ml_benchmark_connect_with" {
  description = "How to connect to each instance"
  value       = { for k, m in module.infra_benchmarks : k => "ssh -i aws_key.pem ubuntu@${m.resource_benchmark_public_ip}" }
}
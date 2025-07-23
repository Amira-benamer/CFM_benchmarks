output "ml_benchmark_public_ips" {
  description = "Public IPs of all ml benchmarking instances"
  value       = { for k, m in module.infra_benchmarks : k => m.resource_benchmark_public_ip }
}
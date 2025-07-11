output "nvidia_benchmark_public_ip" {
  description = "Public IP of the NVIDIA benchmarking instance"
  value       = aws_instance.nvidia_benchmark.public_ip
}

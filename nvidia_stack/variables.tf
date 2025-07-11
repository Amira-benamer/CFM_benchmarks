
variable "region" {
  description = "AWS region to deploy instances"
  type        = string
  default     = "eu-west-1" # or no default if you want to force passing it
}

variable "key_name" {
  description = "EC2 key pair name for SSH access"
  type        = string
}

variable "experiment_name" {
  description = "Name for experiment tagging and logging"
  type        = string
  default     = "nvidia-benchmark"
}

variable "nvidia_instance_type" {
  description = "Instance type for NVIDIA benchmarking (e.g., g5.xlarge, p3.2xlarge)"
  type        = string
  default     = "g5.xlarge"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "vpc_name" {
  description = "Name for the VPC and related resources"
  type        = string
  default     = "ml-benchmark-vpc"
}

variable "public_subnet_cidr" {
  type        = string
  description = "CIDR block for the public subnet"
}

variable "private_subnet_cidr" {
  type        = string
  description = "CIDR block for the private subnet"
}
variable "availability_zone" {
  description = "Availability zone for the subnet"
  type        = string
  default     = "us-east-1a"
}

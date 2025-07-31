variable "ami" {
  description = "AMI ID for the EC2 instance"
  type        = string
}

variable "instance_type" {
  description = "Instance type for the EC2 instance"
  type        = string
  default     = "g4dn.xlarge" # Default value can be overridden
}

variable "name" {
  description = "Name for the instance"
  type        = string
  default     = "nvidia_benchmark" # Default value can be overridden
}

variable "key_name" {
  description = "EC2 key pair name for SSH access"
  type        = string
}

variable "experiment_name" {
  description = "Name for experiment tagging and logging"
  type        = string
  default     = "ml_benchmark_experiment" # Default value can be overridden
}


variable "vpc_id"{
  description = "ID of the VPC"
  type        = string

}

variable "public_subnet_id" {
  description = "ID of the public subnet for the instance"
  type        = string
}

variable "private_subnet_id" {
  description = "ID of the private subnet for the instance"
  type        = string
}

variable "security_group_id" {
  description = "ID of the security group for the instance"
  type        = string
}

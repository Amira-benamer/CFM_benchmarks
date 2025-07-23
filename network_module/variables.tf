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

# variable "subnet_cidr" {
#   description = "CIDR block for the subnet"
#   type        = string
#   default     = "10.0.1.0/24"
# }

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
  default     = "eu-west-1"
}


region            = "eu-west-1"
availability_zone = "eu-west-1a"

# EC2 instance configuration
key_name             = "key-test-cfm"
experiment_name      = "nvidia-benchmark"
nvidia_instance_type = "t3.2xlarge" #"g4dn.xlarge" #"g5.xlarge"

# VPC and subnet configuration
vpc_cidr            = "10.0.0.0/16"
vpc_name            = "ml-benchmark-vpc"
public_subnet_cidr  = "10.0.1.0/24"
private_subnet_cidr = "10.0.2.0/24"

instances = [
  #   {
  #     name           = "infrentia"
  #     ami            = "ami-088e8deff9e0aafeb"
  #     instance_type  = "g4dn.xlarge"
  #     experiment_name = "ml_benchmark_1"
  #   },
  {
    name            = "nvidia"
    # ami             = "ami-088e8deff9e0aafeb" # Ireland
    ami             = "ami-00f71ac70c2d6344d" # Ohio
    instance_type   = "t3.2xlarge"
    experiment_name = "nvidia"
    key_name        = "key-test-cfm"
  }
]

region              = "us-east-2"
availability_zone   = "us-east-2a"
key_name            = "key-test-cfm"
vpc_cidr            = "10.0.0.0/16"
vpc_name            = "ml-benchmark-vpc"
public_subnet_cidr  = "10.0.1.0/24"
private_subnet_cidr = "10.0.2.0/24"

instances = [
    {
      name           = "infrentia"
      ami            = "ami-07613d14174339989" #DLAMI
      instance_type  = "inf2.xlarge"
      experiment_name = "inferentia"
      user_data_file      = "user_data_inf2.sh"
      key_name            = "key-test-cfm"
    },
  {
    name            = "nvidia"
    ami             = "ami-0001fbebb718e22df" #"ami-088e8deff9e0aafeb" 
    instance_type   = "g4dn.xlarge"
    experiment_name = "nvidia-g5"
    user_data_file      = "user_data_nvidia.sh"
    key_name            = "key-test-cfm"
  }
]

region              = "us-east-2"
availability_zone   = "us-east-2a"
key_name            = "key-test-cfm"
vpc_cidr            = "10.0.0.0/16"
vpc_name            = "ml-benchmark-vpc"
public_subnet_cidr  = "10.0.1.0/24"
private_subnet_cidr = "10.0.2.0/24"
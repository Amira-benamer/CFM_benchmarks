
# data "aws_ssm_parameter" "nvidia_ami" {
#   name = "ami-0ae584dd3352860b7"
# }

module "network" {
  source            = "../network_module"
  vpc_cidr          = var.vpc_cidr
  vpc_name          = var.vpc_name
  public_subnet_cidr       = var.public_subnet_cidr
  private_subnet_cidr      = var.private_subnet_cidr
  availability_zone = var.availability_zone
}


resource "aws_instance" "nvidia_benchmark" {
  ami           = "ami-088e8deff9e0aafeb" 
  instance_type = var.nvidia_instance_type
  key_name      = var.key_name

  subnet_id              = module.network.public_subnet_id
  vpc_security_group_ids = [module.network.security_group_id]

  user_data = templatefile("${path.module}/user_data_nvidia.sh.tftpl", {
  experiment_name = var.experiment_name
  track_cost_py   = file("${path.module}/track_cost.py")
  path            = path.module 
  })

  tags = {
    Name     = var.experiment_name
    Platform = "nvidia"
  }
}

module "network" {
  source              = "../network_module"
  vpc_cidr            = var.vpc_cidr
  vpc_name            = var.vpc_name
  public_subnet_cidr  = var.public_subnet_cidr
  private_subnet_cidr = var.private_subnet_cidr
  availability_zone   = var.availability_zone
 
}

module "infra_benchmarks" {
  source   = "../infra_module"
  for_each = { for inst in var.instances : inst.name => inst }

  name              = each.value.name
  instance_type     = each.value.instance_type
  ami               = each.value.ami
  experiment_name   = each.value.experiment_name
  key_name          = each.value.key_name
  vpc_id            = module.network.vpc_id
  public_subnet_id  = module.network.public_subnet_id
  private_subnet_id = module.network.private_subnet_id
  security_group_id = module.network.security_group_id
}
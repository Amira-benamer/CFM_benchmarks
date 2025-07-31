resource "aws_instance" "ml_benchmark" {
  subnet_id     = var.public_subnet_id
  ami           = var.ami
  instance_type = var.instance_type
  key_name      = var.key_name
  vpc_security_group_ids = [var.security_group_id]

  tags = {
    Name     = var.experiment_name
    Platform = var.name
    Project = "13939"
  }
}

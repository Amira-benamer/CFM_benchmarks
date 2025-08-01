


resource "aws_instance" "ml_benchmark" {
  subnet_id     = var.public_subnet_id
  ami           = var.ami
  instance_type = var.instance_type
  key_name      = var.key_name
  vpc_security_group_ids = [var.security_group_id]

  instance_market_options {
    market_type = "spot"

    spot_options {
      spot_instance_type            = "one-time"  # or "persistent"
      instance_interruption_behavior = "terminate"  # or "stop" / "hibernate"
    }
  }
  
  user_data = templatefile("${path.module}/user_data.sh", {
    experiment_name = var.experiment_name
    instance_type   = var.instance_type
  })
  

  tags = {
    Name     = var.experiment_name
    Platform = var.name
  }
}

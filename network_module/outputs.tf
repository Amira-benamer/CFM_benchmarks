# Outputs
output "vpc_id" {
  value = aws_vpc.this.id
}

output "public_subnet_id" {
  value = aws_subnet.public.id
}

output "private_subnet_id" {
  value = aws_subnet.private.id
}

output "security_group_id" {
  value = aws_security_group.this.id
}

output "nat_gateway_id" {
  value = aws_nat_gateway.this.id
}

output "internet_gateway_id" {
  value = aws_internet_gateway.this.id
}

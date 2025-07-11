# resource "aws_iam_role" "ec2_role" {
#   name = "ec2_s3_read_role"

#   assume_role_policy = jsonencode({
#     Version = "2012-10-17"
#     Statement = [{
#       Effect = "Allow"
#       Principal = {
#         Service = "ec2.amazonaws.com"
#       }
#       Action = "sts:AssumeRole"
#     }]
#   })
# }

# resource "aws_iam_policy" "s3_read_policy" {
#   name = "S3ReadTrackCostPolicy"
#   policy = jsonencode({
#     Version = "2012-10-17"
#     Statement = [{
#       Effect = "Allow"
#       Action = ["s3:GetObject"]
#       Resource = ["arn:aws:s3:::cfm-test-benchmark/track_cost.py"]
#     }]
#   })
# }

# resource "aws_iam_role_policy_attachment" "attach_policy" {
#   role       = aws_iam_role.ec2_role.name
#   policy_arn = aws_iam_policy.s3_read_policy.arn
# }

# resource "aws_iam_instance_profile" "ec2_instance_profile" {
#   name = "ec2_instance_profile"
#   role = aws_iam_role.ec2_role.name
# }

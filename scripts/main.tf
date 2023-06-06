terraform {
    required_providers {
        aws = {
            source  = "hashicorp/aws"
            version = "~> 3.0"
        }
    }

    required_version = ">= 0.13"
}

provider "aws" {
    region  = "eu-central-1"
    profile = "Maintainer-7s-sandbox"
}

# G3 will not do, because it has compute capability 5 (we need 7 at least).
# ---------------------------------------------------------------------------------------------------
# || Instance Type | Tesla V100 GPUs | GPU Memory (GB)	| vCPUs	| Memory (GB) | On-Demand Price/hr*	||
# ---------------------------------------------------------------------------------------------------
# || p3.2xlarge	   |  1	             | 16               | 8	    | 61	      | $3.06	            ||
# || p3.8xlarge	   |  4              | 64	            | 32	| 244	      | $12.24	            ||
#----------------------------------------------------------------------------------------------------
resource "aws_instance" "ec2_instance" {
    ami           = "ami-094950f08c57b4f62"
    count         = "1"
    subnet_id     = "subnet-02bbc947d024eaa86"
    instance_type = "p3.2xlarge"
    key_name      = "artem_sereda"
}

## Setup PySpark
```
sudo apt update && sudo apt upgrade
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3-pip
sudo apt install python3.8
sudo apt install awscli
aws configure
pip3 install flintrock boto3 sty
```
set inbound rule for flintrock security group
```
flintrock --config config.yaml run-command SparkCluster "sudo yum update -y && sudo yum install git libcurl python3 -y && pip3 install --user warc3-wet beautifulsoup4 requests numpy scikit-learn"
```

```
./run.sh mnist kmeans 2 10 2 outputs 15618 0 0
```
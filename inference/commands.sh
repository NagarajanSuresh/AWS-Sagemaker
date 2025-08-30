aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 853973692277.dkr.ecr.us-east-1.amazonaws.com

docker build -t ns2312-lr-inference --platform linux/amd64 .

docker tag ns2312-lr-inference:latest 853973692277.dkr.ecr.us-east-1.amazonaws.com/ns2312-lr-inference:latest

docker push 853973692277.dkr.ecr.us-east-1.amazonaws.com/ns2312-lr-inference:latest
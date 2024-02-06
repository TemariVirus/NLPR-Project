CONTAINER_STRING="nlpr"
AWS_ACCOUNT_ID="339712694812"
URL_STRING=".dkr.ecr.us-east-1.amazonaws.com"
IMAGE_STRING="latest"
ECR_IMAGE_URI="$AWS_ACCOUNT_ID$URL_STRING/$CONTAINER_STRING:$IMAGE_STRING"

# sync folder
aws s3 sync s3://nlpr/ ~/awsdocker/

# log in to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID$URL_STRING"

# remove the image to save space
docker image prune -a -f
docker system prune -f

# build image
docker build --no-cache --tag "$CONTAINER_STRING" .

# tag and push to AWS ECR
docker tag $CONTAINER_STRING:latest "$ECR_IMAGE_URI"
docker push "$ECR_IMAGE_URI"

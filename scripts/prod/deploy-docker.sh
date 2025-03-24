echo "Building the image"
docker buildx build --file prod.Dockerfile --platform=linux/amd64 -t sunchhaykhoun/mptc-poc:khmer-ocr-api .

echo "Pushing the image"
docker push sunchhaykhoun/mptc-poc:khmer-ocr-api

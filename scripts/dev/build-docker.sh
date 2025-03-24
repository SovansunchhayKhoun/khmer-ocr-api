echo "Building the image"
docker build --progress=plain -t khmer-ocr-api .

echo "Running the container"
docker run --name khmer-ocr-api -p 3234:3234 khmer-ocr-api
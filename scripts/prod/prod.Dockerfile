ARG PYTHON_VERSION=3.12.4
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /code

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY ./scripts/prod/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

COPY ./start-api.sh .

EXPOSE 3234

CMD ["fastapi", "run", "/code/app/main.py", "--host", "0.0.0.0", "--port", "3234", "--reload"]

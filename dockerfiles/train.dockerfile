FROM python:3.12-slim AS base

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY train_requirements.txt train_requirements.txt

# Copy your preprocessing/training code
COPY src/flowerclassif/train.py src/flowerclassif/train.py

# Install Python dependencies (including ruff)
RUN pip install --verbose -r train_requirements.txt

# Default command to run training
CMD ["python", "src/flowerclassif/train.py"]
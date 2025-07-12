FROM python:3.12-slim AS base

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first to leverage caching
COPY train_requirements.txt train_requirements.txt
COPY pyproject.toml pyproject.toml

# Install dependencies
RUN pip install -r train_requirements.txt --verbose
RUN pip install . --no-deps --verbose

# Now copy source code and data
COPY src/ src/
COPY data/102flowers/raw_images/ data/102flowers/raw_images/
COPY data/labels.csv data/labels.csv

# Preprocess data
RUN python src/flowerclassif/preprocess.py

# Train the model
RUN python src/flowerclassif/train.py

FROM python:3.12-slim AS base

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY preprocess_requirements.txt preprocess_requirements.txt
 
# Copy your preprocessing code
COPY src/flowerclassif/preprocess.py src/flowerclassif/preprocess.py

# Install Python dependencies (including pandas, gcsfs, etc.)
RUN pip install -r preprocess_requirements.txt

# Set default command (adjust path and script name as needed)
CMD ["python", "src/flowerclassif/preprocess.py"]

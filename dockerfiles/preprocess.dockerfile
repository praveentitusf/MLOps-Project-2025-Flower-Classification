# Use official PyTorch image with CUDA support (or CPU-only if you want)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory inside container
WORKDIR /app

# Copy your preprocessing script and any necessary files
COPY gc_preprocess.py ./  # Adjust filename if needed

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (add more if your script needs)
RUN pip install --no-cache-dir \
    torchvision pandas Pillow

# Command to run your preprocessing script
CMD ["python", "preprocess.py"]

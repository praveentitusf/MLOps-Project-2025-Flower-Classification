FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY backend_requirements.txt /app/backend_requirements.txt
COPY src/flowerclassif/backend.py /app/src/flowerclassif/backend.py
COPY src/flowerclassif/train.py /app/src/flowerclassif/train.py
COPY src/flowerclassif/__init__.py /app/src/flowerclassif/__init__.py

RUN pip install --verbose -r backend_requirements.txt

ENV PYTHONPATH=/app

EXPOSE $PORT
CMD exec uvicorn --port $PORT --host 0.0.0.0 src.flowerclassif.backend:app

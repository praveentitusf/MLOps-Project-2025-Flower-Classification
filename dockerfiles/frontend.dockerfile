FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY frontend_requirements.txt /app/frontend_requirements.txt
COPY src/flowerclassif/frontend.py /app/src/flowerclassif/frontend.py

RUN --mount=type=cache,target=/root/.cache/pip pip install -r frontend_requirements.txt

EXPOSE $PORT

ENTRYPOINT ["sh", "-c", "streamlit run src/flowerclassif/frontend.py --server.port $PORT --server.address=0.0.0.0"]

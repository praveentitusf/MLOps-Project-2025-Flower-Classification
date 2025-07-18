from contextlib import asynccontextmanager
import anyio
import torch
import os
import fsspec
import csv
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms
from src.flowerclassif.train import FlowerResNetClassifier
from google.cloud import secretmanager

# Constants
GCS_CKPT_PATH = "gs://mlops-storage/trained_model/latest_model_resnet18.ckpt"
GCS_CSV_PATH = "gs://mlops-storage/data/flower_labels.csv"

LOCAL_CKPT_PATH = "src/flowerclassif/latest_model.ckpt"
LOCAL_CSV_PATH = "src/flowerclassif/flower_labels.csv"

# --------------------------
# Secret Manager Access
# --------------------------
def get_secret(secret_id: str, project_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Optionally set secret as env var
os.environ["MY_SECRET"] = get_secret("secret-1", "mlopsdat")

# --------------------------
# Ensure local caching
# --------------------------
def ensure_local_file(gcs_path: str, local_path: str):
    if not os.path.exists(local_path):
        print(f"Downloading from GCS → {local_path}")
        with fsspec.open(gcs_path, "rb" if gcs_path.endswith(".ckpt") else "r") as src, \
             open(local_path, "wb" if gcs_path.endswith(".ckpt") else "w") as dst:
            dst.write(src.read())

# --------------------------
# FastAPI App with Lifespan
# --------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, transform, class_names

    # Ensure local cache exists
    ensure_local_file(GCS_CKPT_PATH, LOCAL_CKPT_PATH)
    ensure_local_file(GCS_CSV_PATH, LOCAL_CSV_PATH)

    # Load model from checkpoint
    model = FlowerResNetClassifier.load_from_checkpoint(LOCAL_CKPT_PATH)
    model.eval()

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load class names
    with open(LOCAL_CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        class_names = {int(row["label"]): row["Actual_name"] for row in reader}

    yield

    del model, transform, class_names


app = FastAPI(lifespan=lifespan)

# --------------------------
# Prediction logic
# --------------------------
def predict_image(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        predicted_idx = torch.argmax(output).item()

    label = predicted_idx + 1  # Your labels start at 1
    return class_names.get(label, f"Class {label}")


@app.get("/")
async def root():
    return {"message": "Flower Classification API"}


@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        temp_path = file.filename

        async with await anyio.open_file(temp_path, "wb") as f:
            await f.write(contents)

        prediction = predict_image(temp_path)
        return {"Filename": file.filename, "Prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

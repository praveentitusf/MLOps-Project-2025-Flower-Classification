import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
import gcsfs
from io import BytesIO

# -------------------------------
# Paths on GCS
# -------------------------------
bucket = "mlops-storage"
image_dir = f"gs://{bucket}/data/102flowers/raw_images"
label_file = f"gs://{bucket}/data/labels.csv"
output_dir = f"gs://{bucket}/data/102flowers/preprocessed/"

# -------------------------------
# Setup gcsfs
# -------------------------------
fs = gcsfs.GCSFileSystem()

# -------------------------------
# Load Data
# -------------------------------
image_paths = sorted([f for f in fs.ls(image_dir) if f.endswith(".jpg")])

with fs.open(label_file, 'r') as f:
    labels = pd.read_csv(f)['label'].tolist()

assert len(image_paths) == len(labels), "Mismatch between number of images and labels."

data = list(zip(image_paths, labels))
random.seed(42)
random.shuffle(data)

# -------------------------------
# Split Data
# -------------------------------
total = len(data)
train_end = int(0.7 * total)
val_end = int(0.9 * total)
train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# -------------------------------
# Preprocessing Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Save .pt Files TO GCS
# -------------------------------
def save_tensor_dataset_gcs(split_data, split_name):
    split_dir = os.path.join(output_dir, split_name)
    # gcsfs doesn't need to create folders explicitly, but can do mkdir if you want:
    if not fs.exists(split_dir):
        fs.mkdir(split_dir)

    for i, (img_path, label) in enumerate(split_data):
        with fs.open(img_path, 'rb') as f:
            image = Image.open(BytesIO(f.read())).convert("RGB")
        tensor = transform(image)
        sample = {
            'image': tensor,
            'label': label
        }
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(split_dir, f"{img_name}.pt")

        # Save tensor to an in-memory buffer first
        buffer = BytesIO()
        torch.save(sample, buffer)
        buffer.seek(0)

        # Write buffer content to GCS
        with fs.open(save_path, 'wb') as fout:
            fout.write(buffer.read())

    print(f"✅ Saved {len(split_data)} .pt files to {split_name} in GCS.")

save_tensor_dataset_gcs(train_data, "train")
save_tensor_dataset_gcs(val_data, "val")

# -------------------------------
# Save Test Set images + CSV TO GCS
# -------------------------------
test_img_dir = os.path.join(output_dir, "test_images")
if not fs.exists(test_img_dir):
    fs.mkdir(test_img_dir)

test_label_entries = []

for img_path, label in test_data:
    original_fname = os.path.basename(img_path)
    with fs.open(img_path, 'rb') as fsrc, fs.open(os.path.join(test_img_dir, original_fname), 'wb') as fdst:
        fdst.write(fsrc.read())
    test_label_entries.append((original_fname, label))

test_labels_path = os.path.join(output_dir, "test_labels.csv")
with fs.open(test_labels_path, 'w') as f:
    pd.DataFrame(test_label_entries, columns=["filename", "label"]).to_csv(f, index=False)

print("✅ All done! Outputs saved to GCS under:", output_dir)

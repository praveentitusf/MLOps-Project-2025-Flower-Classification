import os
import shutil
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

'''# Corrected paths according to your root
images_dir = "../../data/102flowers/jpg"
csv_file = "../../data/labels.csv"
output_base = "../../data/102flowers/preprocessed"  # fixed missing slash here'''

images_dir = "data/102flowers/jpg"
csv_file = "data/labels.csv"
output_base = "data/102flowers/preprocessed"

os.makedirs(output_base, exist_ok=True)
os.makedirs(os.path.join(output_base, "test_raw"), exist_ok=True)

# Load CSV and add filename column if needed
df = pd.read_csv(csv_file)
df["filename"] = df.index.map(lambda x: f"image_{x+1:05d}.jpg")

# Split into train/val/test
train_val_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['label'])
train_df, val_df = train_test_split(train_val_df, test_size=0.20, random_state=42, stratify=train_val_df['label'])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def process_split(split_df, split_name):
    split_dir = os.path.join(output_base, split_name)
    os.makedirs(split_dir, exist_ok=True)

    labels = []

    for idx, row in split_df.iterrows():
        image_path = os.path.join(images_dir, row["filename"])
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image)

        base_name = os.path.splitext(row["filename"])[0]
        torch.save(tensor, os.path.join(split_dir, f"{base_name}.pt"))

        labels.append(row["label"])

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    torch.save(labels_tensor, os.path.join(split_dir, "labels.pt"))

    print(f"[{split_name}] Saved {len(split_df)} image tensors and labels.")

# Process train and val splits
process_split(train_df, "train")
process_split(val_df, "val")

# Copy test images as raw (no transform)
for filename in test_df["filename"]:
    src = os.path.join(images_dir, filename)
    dst = os.path.join(output_base, "test_raw", filename)
    shutil.copy(src, dst)

# Save test labels as CSV
test_df[["filename", "label"]].to_csv(os.path.join(output_base, "test_labels.csv"), index=False)
print(f"[test_raw] Copied {len(test_df)} test images and saved test_labels.csv.")

def adjust_labels(file_path):
    # Load labels
    labels = torch.load(file_path)

    # Adjust based on type
    if isinstance(labels, torch.Tensor):
        labels = labels - 1
    elif isinstance(labels, list):
        labels = [label - 1 for label in labels]
    else:
        raise TypeError(f"Unsupported label type in {file_path}: {type(labels)}")

    # Overwrite the original file
    torch.save(labels, file_path)
    print(f"Adjusted labels saved back to {file_path}")

# Apply to both train and val labels
adjust_labels(os.path.join(output_base, "train", "labels.pt"))
adjust_labels(os.path.join(output_base, "val", "labels.pt"))

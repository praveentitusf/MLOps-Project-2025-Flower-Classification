import os
import shutil
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

images_dir = "data_copy/102flowers/jpg"
csv_file = "data_copy/labels.csv"
output_base = "data_copy/102flowers/preprocessed"

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

def process_split_low_memory(split_df, split_name):
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
process_split_low_memory(train_df, "train")
process_split_low_memory(val_df, "val")

# Copy test images as raw (no transform)
for filename in test_df["filename"]:
    src = os.path.join(images_dir, filename)
    dst = os.path.join(output_base, "test_raw", filename)
    shutil.copy(src, dst)

# Save test labels as CSV
test_df[["filename", "label"]].to_csv(os.path.join(output_base, "test_labels.csv"), index=False)
print(f"[test_raw] Copied {len(test_df)} test images and saved test_labels.csv.")

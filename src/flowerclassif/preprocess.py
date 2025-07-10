import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
import shutil

# -------------------------------
# Paths
# -------------------------------
image_dir = "data/102flowers/raw_images"
label_file = "data/labels.csv"
output_dir = "data/102flowers/preprocessed/"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Load Data
# -------------------------------
image_paths = sorted(
    [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(".jpg")]
)

labels = pd.read_csv(label_file)['label'].tolist()
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
    transforms.Resize((224, 224)),  # Standard size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Save .pt Files
# -------------------------------
def save_tensor_dataset(split_data, split_name):
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for i, (img_path, label) in enumerate(split_data):
        image = Image.open(img_path).convert("RGB")
        tensor = transform(image)
        sample = {
            'image': tensor,
            'label': label  # <- Keep 1-based if you want, subtract 1 later in training
        }
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(split_dir, f"{img_name}.pt")
        torch.save(sample, save_path)

    print(f"✅ Saved {len(split_data)} .pt files in {split_name}/")

# Save train and val splits
save_tensor_dataset(train_data, "train")
save_tensor_dataset(val_data, "val")

# -------------------------------
# Test Set: Save raw images + CSV
# -------------------------------
test_img_dir = os.path.join(output_dir, "test_images")
os.makedirs(test_img_dir, exist_ok=True)

test_label_entries = []

for img_path, label in test_data:
    original_fname = os.path.basename(img_path)
    shutil.copy(img_path, os.path.join(test_img_dir, original_fname))
    test_label_entries.append((original_fname, label))

test_labels_path = os.path.join(output_dir, "test_labels.csv")
pd.DataFrame(test_label_entries, columns=["filename", "label"]).to_csv(test_labels_path, index=False)

print("✅ All done! Outputs in:", output_dir)


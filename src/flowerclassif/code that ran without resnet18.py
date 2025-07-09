import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

image_folder = 'data/102flowers/preprocessed'  # folder with .pt files
label_csv = 'data/labels.csv'  # CSV with labels only, no filenames

# 1. List all .pt files sorted (assuming consistent ordering)
all_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.pt')])
all_paths = [os.path.join(image_folder, f) for f in all_files]

print(f"Found {len(all_paths)} image files.")

# 2. Load labels from CSV (assumed single column 'label' or just a column)
df = pd.read_csv(label_csv)
labels = [label - 1 for label in df['label'].tolist()]

print(f"Loaded {len(labels)} labels.")

assert len(all_paths) == len(labels), "Number of images and labels must be the same!"

seed = 999  # Make sure seed is defined

# 3. Split indices for train/val/test
indices = list(range(len(labels)))

train_val_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=seed)
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.25,
    stratify=[labels[i] for i in train_val_idx],
    random_state=seed
)

# 4. Prepare splits as lists of (image_path, label)
train_data = [(all_paths[i], labels[i]) for i in train_idx]
val_data = [(all_paths[i], labels[i]) for i in val_idx]
test_data = [(all_paths[i], labels[i]) for i in test_idx]




import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# -------------------------
# Dataset Class
# -------------------------
class FlowerDataset(Dataset):
    def __init__(self, data):
        self.data = data  # list of (path, label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        tensor = torch.load(path).float()  # Just the image tensor
        return tensor, int(label) 

# -------------------------
# Model (MLP)
# -------------------------
class FlowerMLPClassifier(pl.LightningModule):
    def __init__(self, input_size=3*224*224, num_classes=102):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

# -------------------------
# Example Usage
# -------------------------
# Replace with your actual data
from torch.utils.data import DataLoader

train_loader = DataLoader(FlowerDataset(train_data), batch_size=32, shuffle=True)
val_loader   = DataLoader(FlowerDataset(val_data), batch_size=32)
test_loader  = DataLoader(FlowerDataset(test_data), batch_size=32)

model = FlowerMLPClassifier()

trainer = Trainer(max_epochs=10, accelerator='auto')

# Train
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Test
trainer.test(model, dataloaders=test_loader)

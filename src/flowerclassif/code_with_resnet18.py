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


from torchvision.models import resnet18
from torchvision import transforms

class FlowerResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=102):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Replace final FC layer

        # Optional: Define preprocessing if needed later
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        # Normalize the input if needed
        x = self.transform(x)
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



from torch.utils.data import DataLoader

train_loader = DataLoader(FlowerDataset(train_data), batch_size=32, shuffle=True, num_workers=11)
val_loader   = DataLoader(FlowerDataset(val_data), batch_size=32, num_workers=11)
test_loader  = DataLoader(FlowerDataset(test_data), batch_size=32)

model = FlowerResNetClassifier()

# ------------------------------
# 1. Imports
# ------------------------------
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profilers import PyTorchProfiler

# ------------------------------
# 2. WandB Logger and Callbacks
# ------------------------------
wandb_logger = WandbLogger(
    project='flower-classification',
    name='run1',
    log_model=True
)

checkpoint_cb = ModelCheckpoint(
    monitor='val_loss',
    save_top_k=1,
    mode='min'
)

earlystop_cb = EarlyStopping(
    monitor='val_loss',
    patience=3
)

# ------------------------------
# 3. PyTorch Profiler
# ------------------------------
profiler = PyTorchProfiler(
    dirpath="profiling_logs",
    filename="profiler_report",
    export_to_chrome=True,     # Generates .json for Chrome tracing
    record_functions=True,     # Capture autograd-level details
    use_cpu=False              # Use CUDA timings if training on GPU
)

# ------------------------------
# 4. Trainer Setup
# ------------------------------
trainer = Trainer(
    max_epochs=10,
    accelerator='gpu',
    devices=1,
    logger=wandb_logger,
    profiler=profiler,
    callbacks=[checkpoint_cb, earlystop_cb])

# ------------------------------
# 5. Train and Test
# ------------------------------

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

trainer.save_checkpoint("saved_model/latest_model_resnet18")

#from src.flowerclassif.code_with_resnet18 import FlowerResNetClassifier  # Import your model class
#model = FlowerResNetClassifier.load_from_checkpoint("saved_model/model_ran_on_0507")

#
trainer.test(model, dataloaders=test_loader)

#profiler.export_chrome_trace("trace.json")

#print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
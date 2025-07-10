import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profilers import PyTorchProfiler

# -------------------------
# Paths
# -------------------------
train_data = 'data/102flowers/preprocessed/train/'
val_data = 'data/102flowers/preprocessed/val/'


# -------------------------
# Dataset Class
# -------------------------
class FlowerDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith('.pt')
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        sample = torch.load(self.image_paths[idx])
        image = sample['image'].float()
        label = sample['label'] - 1  # subtract 1 here
        return image, label

# -------------------------
# Model
# -------------------------
class FlowerResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=102):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

if __name__ == "__main__":
    # -------------------------
    # Dataloaders
    # -------------------------
    train_loader = DataLoader(FlowerDataset(train_data), batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(FlowerDataset(val_data), batch_size=32, num_workers=4)

    # -------------------------
    # Logger and Callbacks
    # -------------------------
    wandb_logger = WandbLogger(project='flower-classification', name='resnet18-run', log_model=True)

    checkpoint_cb = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
    earlystop_cb = EarlyStopping(monitor='val_loss', patience=3)

    '''profiler = PyTorchProfiler(
        dirpath="profiling_logs",
        filename="profiler_report",
        export_to_chrome=True,
        record_functions=True,
        use_cpu=False
    )'''

    # -------------------------
    # Train
    # -------------------------
    model = FlowerResNetClassifier()

    trainer = Trainer(
        max_epochs=1,
        accelerator='gpu',  # use 'cpu' if no GPU
        devices=1,
        logger=wandb_logger,
        #profiler=profiler,
        callbacks=[checkpoint_cb, earlystop_cb]
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.save_checkpoint("models/latest_model_resnet18.ckpt")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    import os
    import gcsfs
    from pytorch_lightning.loggers import WandbLogger
    from google.cloud import secretmanager
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == "__main__":
    def get_wandb_api_key(secret_id: str, project_id: str) -> str:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")


    # Set the WandB API key as environment variable before WandB starts
    wandb_key = get_wandb_api_key("wandb-api-key", "mlopsdat")
    os.environ["WANDB_API_KEY"] = wandb_key

    # -------------------------
    # GCS Setup
    # -------------------------
    train_data = 'mlops-storage/preprocessed/train/'
    val_data = 'mlops-storage/preprocessed/val/'

    fs = gcsfs.GCSFileSystem(project='mlopsdat')

# -------------------------
# Dataset Class (GCS-Compatible)
# -------------------------
    class FlowerDatasetGCS(Dataset):
        def __init__(self, gcs_path, fs):
            self.fs = fs
            self.image_paths = sorted([
                f for f in self.fs.ls(gcs_path) if f.endswith('.pt')
            ])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            path = self.image_paths[idx]
            with self.fs.open(path, 'rb') as f:
                sample = torch.load(f)
            image = sample['image'].float()
            label = sample['label'] - 1  # adjust label
            return image, label

# -------------------------
# Model
# -------------------------
class FlowerResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=102):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

# -------------------------
# Training Script
# -------------------------
if __name__ == "__main__":
    train_loader = DataLoader(FlowerDatasetGCS(train_data, fs), batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(FlowerDatasetGCS(val_data, fs), batch_size=128, num_workers=0)

    model = FlowerResNetClassifier()

    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        dirpath='checkpoints/',
        filename='best-checkpoint'
    )

    earlystop_cb = EarlyStopping(monitor='val_loss', patience=3)

    wandb_logger = WandbLogger(
        project='flower-classification',
        name='resnet18-run',
        log_model=True
    )

    trainer = Trainer(
        max_epochs=5,
        accelerator='cpu',  
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, earlystop_cb],
        log_every_n_steps=10
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.save_checkpoint("latest_model_resnet18.ckpt")

    checkpoint_path = "latest_model_resnet18.ckpt"
    gcs_path = "mlops-storage/trained_model/latest_model_resnet18.ckpt"

    with open(checkpoint_path, "rb") as local_file:
        with fs.open(gcs_path, "wb") as gcs_file:
            gcs_file.write(local_file.read())

    print(f"Checkpoint uploaded to gs://{gcs_path}")

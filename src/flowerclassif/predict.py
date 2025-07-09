import sys
from PIL import Image
import torch
from torchvision import transforms
from train import FlowerResNetClassifier  # Replace 'your_model_file' with your actual filename (without .py)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if len(sys.argv) != 3:
    print("Usage: python predict.py <image_path> <model_ckpt_path>")
    sys.exit(1)

image_path = sys.argv[1]
model_ckpt_path = sys.argv[2]

# Preprocessing transform (same normalization as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load and preprocess image
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Load model from checkpoint
model = FlowerResNetClassifier.load_from_checkpoint(model_ckpt_path)
model.eval()

# Predict
with torch.no_grad():
    output = model(image_tensor)
    predicted_class = output.argmax(1).item()

print(f"Predicted class index: {predicted_class+1}")

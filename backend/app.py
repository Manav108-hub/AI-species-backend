from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import json

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define FusionNet architecture
class FusionNet(nn.Module):
    def __init__(self, num_classes=5):  # Default to 5 classes
        super(FusionNet, self).__init__()
        # CNN Stream
        self.cnn_stream = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # EfficientNet Stream
        self.effnet = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.effnet.classifier = nn.Identity()
        for param in self.effnet.parameters():
            param.requires_grad = False

        # Fusion Layer - Output 5 classes
        self.fusion = nn.Sequential(
            nn.Linear(512 + 1280, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)  # 5 output units
        )

    def forward(self, x):
        cnn_features = self.cnn_stream(x).flatten(1)
        eff_features = self.effnet(x)
        combined = torch.cat((cnn_features, eff_features), dim=1)
        return self.fusion(combined)

# Load class names and mapping
try:
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
except FileNotFoundError:
    class_names = ["Cani_aure", "Catt_catt", "Hyen_hyen", "Pant_pard", "Rusa_unic"]

try:
    with open("class_mapping.json", "r") as f:
        class_mapping = json.load(f)
except FileNotFoundError:
    class_mapping = {name: name for name in class_names}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = FusionNet(num_classes=len(class_names)).to(device)
try:
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# API Endpoints
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            predicted_class = class_names[output.argmax().item()]

        # Format into human-readable percentages
        formatted_probs = {
            class_mapping[class_names[i]]: round(float(prob) * 100, 4)
            for i, prob in enumerate(probabilities)
        }

        return {
            "predicted_class": class_mapping[predicted_class],
            "probabilities_percent": formatted_probs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "Model is ready", "device": str(device)}
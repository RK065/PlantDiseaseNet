
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError

app = FastAPI(title="Plant Disease Classifier")

# Device for inference
device = torch.device("cpu")

# Exact class order from your training dataset
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# Locate checkpoint
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_mobilenet.pth"

# Recreate architecture
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(CLASS_NAMES)
)

# Load trained weights
state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Use the pretrained model's validation preprocessing
preprocess = models.MobileNet_V2_Weights.DEFAULT.transforms()


@app.get("/")
def home():
    return {"message": "Plant Disease API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = probabilities.max(dim=1)

    index = predicted_index.item()

    return {
        "predicted_class": CLASS_NAMES[index],
        "confidence": round(confidence.item() * 100, 2)
    }

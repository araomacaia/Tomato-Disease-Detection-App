# ==========================================================
# Tomato Disease Classification API
# ==========================================================

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import requests

# ==========================================================
# Create FastAPI app instance
# ==========================================================
app = FastAPI(title="Tomato Disease Classifier API")

# Allow all origins (useful for React JS or React Native)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Model Configuration
# ==========================================================
MODEL_URL = "http://localhost:8501/v1/models/tomato_model:predict"

CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "Tomato_healthy"
]

IMG_HEIGHT = 224
IMG_WIDTH = 224

# ==========================================================
# 1️⃣ Default route (Home)
# ==========================================================
@app.get("/")
def home():
    return {"message": "Tomato Disease Classification API is running successfully!"}

# ==========================================================
# 2️⃣ Prediction route
# ==========================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    image_data = await file.read()

    # Convert image to RGB and resize
    img = Image.open(BytesIO(image_data)).convert("RGB")
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prepare data for TensorFlow Serving
    payload = {"instances": img_array.tolist()}

    # Send image to TensorFlow model server
    response = requests.post(MODEL_URL, json=payload)
    result = response.json()

    # Extract predictions
    if "predictions" in result:
        preds = np.array(result["predictions"][0])
    else:
        preds = np.array(list(result.values())[0][0])

    # Get top predicted class
    class_id = np.argmax(preds)
    confidence = float(np.max(preds))

    return {
        "predicted_class": CLASS_NAMES[class_id],
        "confidence": round(confidence * 100, 2)
    }
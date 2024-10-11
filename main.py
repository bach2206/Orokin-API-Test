# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import cv2
import asyncio

from utils import preprocess_image, detect_nodule_preprocessing

app = FastAPI(title="Lung Nodule Detection API", version="1.0")

# Configure CORS (Adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
@app.on_event("startup")
def load_ai_model():
    global model
    try:
        model = load_model("orokinv1.h5")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/detect-nodule")
async def detect_nodule(file: UploadFile = File(...), center_x: float = 256.0, center_y: float = 256.0, diameter: float = 50.0):
    """
    Endpoint to detect nodules in a CT scan image.

    Args:
        file (UploadFile): CT scan image file (e.g., PNG, JPEG).
        center_x (float): X-coordinate of the nodule center (optional, default=256.0).
        center_y (float): Y-coordinate of the nodule center (optional, default=256.0).
        diameter (float): Diameter of the nodule (optional, default=50.0).

    Returns:
        JSONResponse: Detection result with probability.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        # Read image data
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_np = np.array(image)

        # Preprocess the image
        preprocessed_image = preprocess_image(image_np)
        preprocessed_image = detect_nodule_preprocessing(preprocessed_image)

        # If your model expects additional features like center and diameter, incorporate them here
        # For simplicity, we'll assume the model only takes the image

        # Expand dimensions to match model input (1, 512, 512, 1)
        input_image = np.expand_dims(preprocessed_image, axis=0)

        # Make prediction
        predictions = model.predict(input_image)
        probability = float(predictions[0][0])

        # Define a threshold for nodule detection
        threshold = 0.9
        nodule_detected = probability >= threshold

        return JSONResponse(content={
            "nodule_detected": nodule_detected,
            "probability": probability
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

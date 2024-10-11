# utils.py

import numpy as np
import cv2
from sklearn.cluster import KMeans

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the input image to match the model's training conditions.
    
    Steps:
    1. Normalize the image.
    2. Resize to 512x512 if not already.
    3. Apply CLAHE.
    4. Expand dimensions to match model input.
    
    Args:
        image (np.ndarray): Input image in grayscale.
    
    Returns:
        np.ndarray: Preprocessed image ready for prediction.
    """
    # Ensure image is in grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize the image to range [0, 255]
    image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Resize to 512x512
    image_resized = cv2.resize(image_norm, (512, 512))
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_clahe = clahe.apply(image_resized)
    
    # Normalize to [0,1]
    image_normalized = image_clahe / 255.0
    
    # Expand dimensions to match model input (512, 512, 1)
    image_final = np.expand_dims(image_normalized, axis=-1)
    
    return image_final

def make_mask(img, center, diam):
    """
    Create a circular mask on the image.
    
    Args:
        img (np.ndarray): Input image.
        center (tuple): (x, y) coordinates for the center.
        diam (int): Diameter of the circle.
    
    Returns:
        np.ndarray: Masked image.
    """
    mask = np.zeros_like(img, dtype=np.uint8)
    mask = cv2.circle(mask, (abs(int(center[0])), abs(int(center[1]))), int(abs(diam // 2)), 255, -1)
    return mask

def detect_nodule_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Additional preprocessing steps if required based on your notebook.
    Modify this function based on specific needs from your notebook.
    
    Args:
        image (np.ndarray): Preprocessed image.
    
    Returns:
        np.ndarray: Final image ready for prediction.
    """
    # Placeholder for additional preprocessing if needed
    return image

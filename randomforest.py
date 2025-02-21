import os
import io
import cv2
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from google.cloud import vision

# Initialize Google Vision API client
try:
    vision_client = vision.ImageAnnotatorClient()
except Exception as e:
    raise RuntimeError(f"Failed to initialize Google Vision API client: {e}")

# Helper Functions
def calculate_contrast(image, bbox):
    """
    Calculate contrast for a bounding box in the image.

    Args:
        image (ndarray): OpenCV image.
        bbox (tuple): Bounding box (x, y, width, height).

    Returns:
        float: Contrast value.
    """
    try:
        x, y, width, height = bbox
        cropped = image[y:y+height, x:x+width]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        contrast = gray.max() - gray.min()
        return contrast
    except Exception as e:
        raise ValueError(f"Error calculating contrast: {e}")

def extract_text_features(image_path: str) -> pd.DataFrame:
    """
    Extract text features using Google Vision API and calculate additional features.
    """
    required_columns = ["x", "y", "width", "height", "AspectRatio", "Area", "Contrast", "text"]
    
    try:
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)

        response = vision_client.text_detection(image=image)
        annotations = response.text_annotations

        img_cv2 = cv2.imread(image_path)
        features = []

        if annotations:
            for annotation in annotations[1:]: 
                bounding_poly = annotation.bounding_poly
                x_min = min([vertex.x for vertex in bounding_poly.vertices])
                y_min = min([vertex.y for vertex in bounding_poly.vertices])
                x_max = max([vertex.x for vertex in bounding_poly.vertices])
                y_max = max([vertex.y for vertex in bounding_poly.vertices])

                width = x_max - x_min
                height = y_max - y_min
                aspect_ratio = width / height if height != 0 else 0
                area = width * height
                contrast = calculate_contrast(img_cv2, (x_min, y_min, width, height))

                features.append({
                    "x": x_min,
                    "y": y_min,
                    "width": width,
                    "height": height,
                    "AspectRatio": aspect_ratio,
                    "Area": area,
                    "Contrast": contrast,
                    "text": annotation.description
                })
        return pd.DataFrame(features, columns=required_columns)

    except Exception as e:
        # Return empty DataFrame with required columns on error
        return pd.DataFrame(columns=required_columns)

    except Exception as e:
        raise RuntimeError(f"Error extracting text features from {image_path}: {e}")

def classify_text(features_df: pd.DataFrame, model_path: str) -> str:
    """
    Classify text using a Random Forest model.

    Args:
        features_df (pd.DataFrame): DataFrame with text features.
        model_path (str): Path to the pre-trained Random Forest model.

    Returns:
        str: Extracted overlay text.
    """
    try:
        if features_df.empty:
            return "NONE"
        # Load the pre-trained Random Forest model
        model = joblib.load(model_path)

        # Ensure required features are present
        required_features = ["x", "y", "width", "height", "AspectRatio", "Area", "Contrast"]
        missing_features = [feature for feature in required_features if feature not in features_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Predict overlay text
        features_df["Prediction"] = model.predict(features_df[required_features])
        overlay_text = features_df[features_df["Prediction"] == 1]["text"].unique()
        return " ".join(overlay_text)

    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}. Please provide a valid path.")
    except ValueError as ve:
        raise ValueError(f"Feature mismatch error: {ve}")
    except Exception as e:
        raise RuntimeError(f"An error occurred during text classification: {e}")
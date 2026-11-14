import os
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import (
    hog,
    local_binary_pattern,
    graycomatrix,
    graycoprops,
)

# ==========================================================
# Load Saved Model
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "saved_models" / "Best_BrainTumor_Model.pkl"
SCALER_PATH = BASE_DIR / "saved_models" / "Scaler.pkl"
ENCODER_PATH = BASE_DIR / "saved_models" / "LabelEncoder.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

print("Models loaded successfully.")

# ==========================================================
# Image Parameters
# ==========================================================

IMG_SIZE = 128

RADIUS = 2
N_POINTS = 8 * RADIUS

# ==========================================================
# Image Preprocessing
# ==========================================================

def preprocess_image(image_path):
    """
    Read image, resize and convert to grayscale.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray


# ==========================================================
# HOG Features
# ==========================================================

def extract_hog(image):

    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


# ==========================================================
# LBP Features
# ==========================================================

def extract_lbp(image):

    lbp = local_binary_pattern(
        image,
        N_POINTS,
        RADIUS,
        method="uniform",
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, N_POINTS + 3),
        range=(0, N_POINTS + 2),
    )

    hist = hist.astype("float")

    hist /= (hist.sum() + 1e-7)

    return hist


# ==========================================================
# GLCM Features
# ==========================================================

def extract_glcm(image):

    glcm = graycomatrix(
        image,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
    )

    contrast = graycoprops(glcm, "contrast")[0, 0]
    dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    asm = graycoprops(glcm, "ASM")[0, 0]

    return np.array([
        contrast,
        dissimilarity,
        homogeneity,
        energy,
        correlation,
        asm,
    ])


# ==========================================================
# Feature Extraction
# ==========================================================

def extract_features(image_path):

    image = preprocess_image(image_path)

    hog_features = extract_hog(image)

    lbp_features = extract_lbp(image)

    glcm_features = extract_glcm(image)

    return np.concatenate([
        hog_features,
        lbp_features,
        glcm_features,
    ])


# ==========================================================
# Prepare Image
# ==========================================================

def prepare_single_image(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} not found.")

    features = extract_features(image_path)

    features = features.reshape(1, -1)

    features = scaler.transform(features)

    return features


# ==========================================================
# Prediction
# ==========================================================

def predict_image(image_path):

    features = prepare_single_image(image_path)

    prediction = model.predict(features)

    probabilities = model.predict_proba(features)[0]

    predicted_class = encoder.inverse_transform(prediction)[0]

    confidence = float(np.max(probabilities) * 100)

    probability_dict = {
        class_name: round(float(prob) * 100, 2)
        for class_name, prob in zip(
            encoder.classes_,
            probabilities,
        )
    }

    return predicted_class, confidence, probability_dict


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    test_image = "test.jpg"   # Replace with an image path

    prediction, confidence, probabilities = predict_image(test_image)

    print("=" * 40)
    print("Prediction Result")
    print("=" * 40)

    print(f"Predicted Class : {prediction}")
    print(f"Confidence      : {confidence:.2f}%")

    print("\nClass Probabilities")
    print("-" * 30)

    for cls, prob in probabilities.items():
        print(f"{cls.capitalize():12s}: {prob:.2f}%")
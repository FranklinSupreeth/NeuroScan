"""
NeuroScan - Brain Tumor MRI Classification
Flask web application (HOG + LBP + GLCM features -> SVM classifier)
"""

import os
import uuid

from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from predictor import predict_image

# ==========================================================
# App configuration
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
SAMPLE_FOLDER = os.path.join(BASE_DIR, "static", "samples")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
MAX_UPLOADS_KEPT = 50  # keep the uploads folder from growing forever

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASS_INFO = {
    "glioma": {
        "label": "Glioma",
        "summary": "A tumour arising from glial cells of the brain or spine.",
    },
    "meningioma": {
        "label": "Meningioma",
        "summary": "A usually slow-growing tumour of the meninges, the membranes covering the brain.",
    },
    "notumor": {
        "label": "No Tumor",
        "summary": "No tumour pattern was detected in this scan.",
    },
    "pituitary": {
        "label": "Pituitary Tumor",
        "summary": "A growth in the pituitary gland at the base of the brain.",
    },
}

SAMPLES = ["glioma", "meningioma", "notumor", "pituitary"]


# ==========================================================
# Helpers
# ==========================================================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def prune_uploads():
    """Delete the oldest uploads so the disk does not fill up."""
    try:
        files = [
            os.path.join(UPLOAD_FOLDER, f)
            for f in os.listdir(UPLOAD_FOLDER)
            if not f.startswith(".")
        ]
        files.sort(key=os.path.getmtime)
        for old in files[:-MAX_UPLOADS_KEPT]:
            os.remove(old)
    except OSError:
        pass


def build_response(filepath, image_url):
    prediction, confidence, probabilities = predict_image(filepath)
    key = str(prediction)
    info = CLASS_INFO.get(key, {"label": key.title(), "summary": ""})
    return {
        "success": True,
        "prediction": key,
        "label": info["label"],
        "summary": info["summary"],
        "confidence": round(float(confidence), 2),
        "probabilities": {str(k): float(v) for k, v in probabilities.items()},
        "image": image_url,
    }


# ==========================================================
# Routes
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html", samples=SAMPLES, classes=CLASS_INFO)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "app": "NeuroScan"})


@app.route("/predict", methods=["POST"])
def predict():
    # Sample image from the gallery
    sample = request.form.get("sample")
    if sample:
        if sample not in SAMPLES:
            return jsonify({"success": False, "message": "Unknown sample."}), 400
        path = os.path.join(SAMPLE_FOLDER, f"{sample}.jpg")
        try:
            return jsonify(build_response(path, url_for("static", filename=f"samples/{sample}.jpg")))
        except Exception as e:  # noqa: BLE001
            return jsonify({"success": False, "message": str(e)}), 500

    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image uploaded."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "message": "Please choose an MRI image."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "message": "Unsupported file format. Use PNG, JPG, BMP or TIFF.",
        }), 400

    try:
        ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        prune_uploads()
        return jsonify(build_response(filepath, url_for("static", filename=f"uploads/{filename}")))
    except Exception as e:  # noqa: BLE001
        return jsonify({"success": False, "message": str(e)}), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify({"success": False, "message": "File too large. Maximum size is 10 MB."}), 413


# ==========================================================
# Local run
# ==========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

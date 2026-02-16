import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# =========================
# Load Model Safely
# =========================
MODEL_PATH = "model.h5"

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Error loading model:", e)
else:
    print("❌ model.h5 not found")

# =========================
# Home Route
# =========================
@app.route("/")
def index():
    return render_template("index.html")

# =========================
# Prediction Route
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model not loaded. Check model.h5"

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Preprocess Image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        result = "Fake"
        confidence = prediction * 100
    else:
        result = "Real"
        confidence = (1 - prediction) * 100

    return render_template(
        "result.html",
        result=result,
        confidence=f"{confidence:.2f}",
        image_path=filepath
    )

# =========================
# Run App (Render Compatible)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

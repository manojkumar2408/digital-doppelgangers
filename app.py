import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# ==============================
# Load trained model
# ==============================
MODEL_PATH = "model.h5"   # change if your model name is different
model = load_model(MODEL_PATH)

# ==============================
# Upload folder
# ==============================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==============================
# Home Page
# ==============================
@app.route("/")
def index():
    return render_template("index.html")


# ==============================
# Prediction Route
# ==============================
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    # Save file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # ==============================
    # Preprocess image
    # ==============================
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # ==============================
    # Prediction
    # ==============================
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        result = "Real"
        confidence = prediction * 100
        color = "green"
    else:
        result = "Fake"
        confidence = (1 - prediction) * 100
        color = "red"

    return render_template(
        "result.html",
        result=result,
        confidence=round(confidence, 2),
        image_path=filepath,
        color=color
    )


# ==============================
# Run App (Render Compatible)
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port, debug=False)

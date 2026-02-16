import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------------------------------
# App configuration
# -------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------
# Load trained model
# -------------------------------
MODEL_PATH = "model.h5"   # change if your model name differs
model = load_model(MODEL_PATH)

IMG_SIZE = (224, 224)  # change if your model uses different size


# -------------------------------
# Home page
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -------------------------------
# Prediction route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("result.html", result="No file uploaded")

    file = request.files["file"]

    if file.filename == "":
        return render_template("result.html", result="No file selected")

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

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


# -------------------------------
# Run app (for local testing)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

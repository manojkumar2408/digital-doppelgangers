from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import random   # temporary prediction (replace with model later)

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # 🔹 Temporary prediction (replace with CNN model)
    result = random.choice(["Real", "Fake"])
    confidence = round(random.uniform(85, 99), 2)

    return render_template("result.html",
                           prediction=result,
                           confidence=confidence,
                           filename=filename)

if __name__ == "__main__":
    app.run(debug=True)

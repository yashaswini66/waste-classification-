import os
from flask import Flask, request, render_template, jsonify, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Path to save uploaded files
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
MODEL_PATH = r"C:\Users\yashu\Desktop\waste classification\my_model.h5"
model = load_model(MODEL_PATH)

# Class labels
CLASS_NAMES = ['Organic', 'Recycle', 'Unknown']

# Helper function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Route for the main page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = round(100 * np.max(predictions), 2)

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": f"{confidence}%",
            "file_path": file_path
        })

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)

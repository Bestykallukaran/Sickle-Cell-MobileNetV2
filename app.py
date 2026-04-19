from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import cv2
import os
import uuid

app = Flask(__name__)

# 📂 Upload folder
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Allowed file types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# -------------------------------
# 🧠 Load Model
# -------------------------------
try:
    model = tf.keras.models.load_model("sickle_model.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# -------------------------------
# 🔍 Check file type
# -------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------
# 🧪 Prediction Function
# -------------------------------
def predict_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return "Invalid Image", 0, "error"

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)) / 255.0

        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]

        # Adjust based on your training class_indices
        # Usually: {'Negatives': 0, 'Positives': 1}
        if pred > 0.5:
            return "Sickle Cell Detected", round(float(pred * 100), 2), "positive"
        else:
            return "Normal Cell", round(float((1 - pred) * 100), 2), "negative"

    except Exception as e:
        return f"Error: {str(e)}", 0, "error"

# -------------------------------
# 🏠 Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# -------------------------------
# 📤 Upload & Predict
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'})

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'})

    # Save file safely
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Predict
    label, confidence, status = predict_image(filepath)

    return jsonify({
        'label': label,
        'confidence': confidence,
        'status': status,
        'image_url': f'/uploads/{filename}'
    })

# -------------------------------
# 📷 Serve uploaded images
# -------------------------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# -------------------------------
# 🚀 Run App
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)
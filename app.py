from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

# ─── Load model once at startup ───────────────────────
model = tf.keras.models.load_model("sickle_model.h5")
print("✅ Model loaded!")

def predict_image(img_path):
    img     = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (224, 224)) / 255.0
    pred    = model.predict(np.expand_dims(resized, axis=0), verbose=0)[0][0]
    
    # Adjust this based on your class_indices output!
    # If class_indices = {'Negatives': 0, 'Positives': 1}
    if pred > 0.5:
        label      = "Sickle Cell Detected"
        confidence = pred * 100
        status     = "positive"
    else:
        label      = "Normal Cell"
        confidence = (1 - pred) * 100
        status     = "negative"
    
    return label, round(float(confidence), 2), status

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Save uploaded file
    filename  = f"{uuid.uuid4().hex}_{file.filename}"
    filepath  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Predict
    label, confidence, status = predict_image(filepath)
    
    return jsonify({
        'label':      label,
        'confidence': confidence,
        'status':     status,
        'image_url':  f'/uploads/{filename}'
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
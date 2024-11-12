from flask import Flask, request, jsonify
import librosa
import os
import numpy as np
import pickle
from io import BytesIO
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model and RFE selector
with open('model/dog_bark_classifier33.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/rfe_selector.pkl', 'rb') as rfe_file:
    rfe = pickle.load(rfe_file)

# Function to extract features from audio
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return np.concatenate((np.mean(mfccs.T, axis=0), 
                           np.mean(chroma.T, axis=0),
                           np.mean(spectral_contrast.T, axis=0)))

# Default route to show backend is running
@app.route('/')
def index():
    return "<h1>Backend is running</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    if 'audiofile' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['audiofile']
    print(f"Received file: {file.filename}, MIME type: {file.mimetype}")

    if file and file.mimetype == 'audio/x-wav':
        # Use in-memory processing
        y, sr = librosa.load(BytesIO(file.read()), sr=22050)
        
        # Feature extraction and prediction
        features = extract_features(y, sr)
        features_rfe = rfe.transform(features.reshape(1, -1))
        prediction = model.predict(features_rfe)[0]

        return jsonify({"prediction": prediction})

    return jsonify({"error": "Only .wav files are accepted"}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

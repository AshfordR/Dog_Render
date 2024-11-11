# from flask import Flask, render_template, request, redirect, url_for
# import librosa
# import numpy as np
# import pickle
# import os
# from werkzeug.utils import secure_filename

# # Initialize Flask app
# app = Flask(__name__)

# # Load the trained model and RFE selector
# with open('model/dog_bark_classifier33.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# with open('model/rfe_selector.pkl', 'rb') as rfe_file:
#     rfe = pickle.load(rfe_file)

# # Function to extract features from audio
# def extract_features(y, sr):
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr)
#     spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
#     return np.concatenate((np.mean(mfccs.T, axis=0), 
#                            np.mean(chroma.T, axis=0),
#                            np.mean(spectral_contrast.T, axis=0)))

# # Home page route
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     print("Received method:", request.method)  # Debugging line
#     if 'audiofile' not in request.files:
#         return redirect(request.url)

#     file = request.files['audiofile']
#     if file and file.filename.endswith('.wav'):
#         # Save the file to a secure location
#         filename = secure_filename(file.filename)
#         filepath = os.path.join('uploads', filename)
#         file.save(filepath)

#         # Load the audio and extract features
#         y, sr = librosa.load(filepath, sr=22050)
#         features = extract_features(y, sr)
#         features_rfe = rfe.transform(features.reshape(1, -1))

#         # Predict the class
#         prediction = model.predict(features_rfe)[0]
        
#         return render_template('result.html', prediction=prediction)

#     return redirect(url_for('index'))

# if __name__ == '__main__':
#     # Ensure uploads folder exists
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(debug=True)


































from flask import Flask, request, jsonify
import librosa
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
    
    # Log file details for debugging
    print(f"Received file: {file.filename}, MIME type: {file.mimetype}")

    # Check MIME type to confirm it's a .wav file
    if file and file.mimetype == 'audio/x-wav':
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        y, sr = librosa.load(filepath, sr=22050)
        features = extract_features(y, sr)
        features_rfe = rfe.transform(features.reshape(1, -1))

        prediction = model.predict(features_rfe)[0]

        return jsonify({"prediction": prediction})

    return jsonify({"error": "Only .wav files are accepted"}), 400

if __name__ == '__main__':
    # Ensure uploads folder exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, host='0.0.0.0', port=5000)  # Change host to '0.0.0.0'

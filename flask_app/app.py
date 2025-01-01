from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('../saved_models/emotion_recognition_model.h5')

# Define emotions
emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised']

# Path to the sample image
SAMPLE_IMAGE_PATH = os.path.join('static', 'sample.jpg')

def resize_sample_image(input_path, output_path, size=(300, 300)):
    image = cv2.imread(input_path)
    resized_image = cv2.resize(image, size)
    cv2.imwrite(output_path, resized_image)


# Helper function for prediction
def predict_emotion(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    emotion_index = np.argmax(prediction)
    return emotions[emotion_index]

# Predict emotion for the sample image
sample_prediction = predict_emotion(SAMPLE_IMAGE_PATH)

@app.route('/')
def index():
    return render_template('index.html', sample_prediction=sample_prediction)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No file selected!', 400

    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    # Predict emotion
    prediction = predict_emotion(filepath)

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    SAMPLE_IMAGE_PATH = os.path.join('static', 'sample.jpg')
    RESIZED_SAMPLE_IMAGE_PATH = os.path.join('static', 'sample_resized.jpg')
    resize_sample_image(SAMPLE_IMAGE_PATH, RESIZED_SAMPLE_IMAGE_PATH)
    
    app.run(debug=True)


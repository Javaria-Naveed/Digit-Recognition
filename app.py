from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import os

app = Flask(__name__)
model = load_model('mnist_model_mine.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Predict digit based on background color
    background_color = predict_background(file_path, 0.5)
    if background_color:
        predicted_digit = predict_digit(file_path)
    else:
        negative_file_path = os.path.join('uploads', 'neg_' + file.filename)
        take_negative(file_path, negative_file_path)
        predicted_digit = predict_digit(negative_file_path)

    clear_uploads_directory()

    return jsonify({'image_url': file_path, 'predicted_digit': predicted_digit})

def predict_digit(image_path):
    # Load the image and preprocess it
    img = image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the pixel values to be between 0 and 1

    # Make the prediction
    predictions = model.predict(img_array)
    # digit = np.argmax(predictions)
    digit = int(np.argmax(predictions))  # Convert to regular Python integer
    return digit

def take_negative(input_path, output_path):
    # Open the image
    img = Image.open(input_path).convert('L')  # Convert to grayscale if not already
    
    # Take negative of the image
    negative_img = Image.eval(img, lambda x: 255 - x)
  
    # Save the negative image
    negative_img.save(output_path)

def predict_background(image_path, threshold=0.5):
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))  # Resize image for analysis

    # Count the number of black pixels
    num_black_pixels = np.sum(image == 0)

    # Determine if the background is black based on the threshold
    is_black_background = num_black_pixels / image.size > threshold

    return is_black_background

UPLOADS_FOLDER = 'uploads'

def clear_uploads_directory():
    for file_name in os.listdir(UPLOADS_FOLDER):
        file_path = os.path.join(UPLOADS_FOLDER, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    app.run(debug=True)

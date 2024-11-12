import numpy as np
import cv2
from PIL import Image
import os   
from tensorflow.keras.models import load_model
from HandwritingDetection import HEIGHT, WIDTH

model = load_model('best_handwriting_model.keras')
print(model.summary())



def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')  # Open image and convert to RGB
    img = img.resize((WIDTH, HEIGHT))          # Resize to match model's input size
    img_array = np.array(img).astype('float32') / 255.0  # Normalize to [0, 1]
    img_array = img_array.reshape(1, WIDTH, HEIGHT, 3)   # Add batch dimension
    return img_array

def predict_image(model, img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    label = np.argmax(prediction, axis=1)[0]  # Get index of the highest probability
    return "Handwriting" if label == 1 else "Printed"

# Example usage:
test_path = '/Users/tamim028/github/Python/Deep Learning/HandwritingRecognition/TestImages'

for filename in os.listdir(test_path):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(test_path, filename)
        result = predict_image(model, img_path)
        print(f"Image: {filename} - Prediction: {result}")
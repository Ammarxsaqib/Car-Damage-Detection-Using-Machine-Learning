import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Define function for preprocessing image
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize image
    return image

# Load the trained model
model_path = 'C:/Users/moham/OneDrive/Desktop/CV_Project/car_intensity3.h5'
model = load_model(model_path)

# Define image path for prediction
image_path = 'C:/Users/moham/OneDrive/Desktop/CV_Project/damaged.jpg'

# Preprocess the image
preprocessed_image = preprocess_image(image_path)

# Make prediction
prediction = model.predict(preprocessed_image)

# Decode prediction
if prediction[0][0] > prediction[0][1]:
    print("Predicted label: Minor")
else:
    print("Predicted label: Severe")

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Define the path to the saved model
model_path = 'C:/Users/moham/OneDrive/Desktop/CV_Project/car_observation.h5'

# Load the saved model
model = load_model(model_path)

# Function to preprocess the image
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict the label
def predict_image(img_path):
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)
    label = 'Not Damaged' if prediction[0] > 0.5 else 'Damaged'
    return label

# Example usage
img_path = 'C:/Users/moham/OneDrive/Desktop/CV_Project/damaged.jpg'  # Replace with the path to your image
label = predict_image(img_path)
print(f'The predicted label for the image is: {label}')
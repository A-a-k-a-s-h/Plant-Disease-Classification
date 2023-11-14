import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from efficientnet import EfficientNetB0

# Load the EfficientNetB0 model with the Plant Disease classification weights
model = EfficientNetB0(weights='plant_disease_classification')

# Preprocess the image
def preprocess_image(image):
    """Preprocess the image for classification."""

    # Resize the image to 256x256
    image = image.resize((256, 256))

    # Normalize the image
    image = image / 255.0

    # Convert the image to a NumPy array
    image = np.asarray(image)

    # Return the preprocessed image
    return image

# Classify the image
def classify_image(image):
    """Classify the image using the EfficientNetB0 model."""

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make a prediction
    prediction = model(preprocessed_image)

    # Get the predicted class label
    class_label = prediction.argmax(axis=1)

    # Return the predicted class label
    return class_label

# Main function
def main():
    """Classify a plant disease image using the EfficientNetB0 model."""

    # Display the app title
    st.title('Plant Disease Classification')

    # Upload the image file
    uploaded_file = st.file_uploader('Upload a plant disease image')

    # If an image is uploaded, classify it
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)

        # Classify the image
        class_label = classify_image(image)

        # Display the prediction
        st.write(f'Predicted class: {class_label}')

if __name__ == '__main__':
    main()

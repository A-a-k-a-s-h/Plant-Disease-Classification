import streamlit as st
from PIL import Image
from PIL.Image import Resampling
import torch
import numpy as np

# Load the pre-trained model
model = torch.load('plant-disease-model.pth',map_location=torch.device('cpu'))


# Define a function to preprocess the image
def preprocess_image(image):
    """Preprocess the image for classification."""

    # Upgrade to a newer version of Pillow if necessary.
    if Pillow.__version__ < '9.1.0':
        raise RuntimeError('Pillow version must be at least 9.1.0.')

    # Check the type of the `image.size` property.
    image_size = image.size
    if type(image_size) is int:
        image_size = (image_size, image_size)

    # Check the type of the `Resampling.BILINEAR` constant.
    resampling_bilinear = Resampling.BILINEAR
    if type(resampling_bilinear) is int:
        resampling_bilinear = str(resampling_bilinear)

    # Resize the image to 256x256.
    image = image.resize(list(image_size), resampling_bilinear)

    # Convert the image to a NumPy array.
    image = np.asarray(image)

    # Normalize the image.
    image = image / 255.0

    # Return the preprocessed image.
    return image

# Define a function to classify the image
def classify_image(image):
    """Classify the image using the pre-trained model."""

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make a prediction
    prediction = model(preprocessed_image)

    # Get the predicted class label
    class_label = prediction.argmax(axis=1)

    # Return the predicted class label
    return class_label

# Define the main function
def main():
    """Classify a plant disease image using the pre-trained model."""

    # Display the app title
    st.title('Plant Disease Classification')

    # Upload the image file
    uploaded_file = st.file_uploader('Upload a plant disease image')

    # If an image is uploaded, classify it
    if uploaded_file is not None:
        # Preprocess the image
        image = Image.open(uploaded_file)
        preprocessed_image = preprocess_image(image)

        # Make a prediction
        prediction = classify_image(preprocessed_image)

        # Display the prediction
        st.write(f'Predicted class: {prediction}')

if __name__ == '__main__':
    main()

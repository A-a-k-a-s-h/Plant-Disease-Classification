import streamlit as st
from PIL import Image
import torch

# Load the pre-trained model
model = torch.load('plant-disease-model.pth')
model.eval()

# Define a function to preprocess the image
def preprocess_image(image):
    """Preprocess the image for classification."""

    # Resize the image to 256x256
    image = image.resize((256, 256), Image.ANTIALIAS)

    # Convert the image to a PyTorch tensor
    image = torch.from_numpy(image)

    # Normalize the image
    image = image / 255.0

    # Return the preprocessed image
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

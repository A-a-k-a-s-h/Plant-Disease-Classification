# main.py
import streamlit as st
from PIL import Image
import torch
from io import BytesIO
import base64

# Load the trained model from the .pth file
model = torch.load('/plant-disease-model.pth', map_location=torch.device('cuda'))
model.eval()

# Function to classify an image
def classify_image(image):
    # Preprocess the image (resize, normalize, convert to tensor, etc.)
    # Pass the preprocessed image through the model
    # Return the predicted class using the predict_image function

# Function to predict image using the model
  def predict_image(image):
      # Load the image, preprocess it, and get the predicted class
      pil_image = Image.open(image)
      class_name = predict_image(pil_image, model)  # Call your existing predict_image function
      return class_name

st.title('Plant Disease Classification App')

# Upload image through Streamlit
uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Classify the uploaded image
    class_name = classify_image(image)
    st.write(f'Predicted class: {class_name}')

# Bar graph report
st.subheader('Classification Accuracy Report')
# Create a bar graph with class names and corresponding accuracies

# Download report as PDF
st.subheader('Download Report as PDF')
st.markdown('Click the button below to generate a PDF report.')

def download_report():
    # Generate the bar graph and save it as a PDF
    # Create a BytesIO object to store the PDF
    output = BytesIO()

    # Create the bar graph
    # (You can use Matplotlib for creating the graph)

    # Save the BytesIO object as a PDF
    # Provide a download link for the user

if st.button('Generate PDF Report'):
    download_report()

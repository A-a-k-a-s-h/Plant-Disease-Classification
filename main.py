# main.py
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Define the model architecture
class PlantDiseaseModel(nn.Module):
    # Your model architecture definition here

# Load the trained model
model = PlantDiseaseModel()
model.load_state_dict(torch.load('/plant-disease-model.pth', map_location=torch.device('cuda')))
model.eval()

# Function to classify an image
def classify_image(image):
    # Preprocess the image (resize, normalize, convert to tensor, etc.)
    # Pass the preprocessed image through the model
    # Return the predicted class

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
    plt.savefig(output, format='pdf')
    output.seek(0)

    # Provide a download link for the user
    b64 = base64.b64encode(output.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="classification_report.pdf">Download PDF Report</a>'
    st.markdown(href, unsafe_allow_html=True)

if st.button('Generate PDF Report'):
    download_report()

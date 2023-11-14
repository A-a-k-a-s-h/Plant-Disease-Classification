import streamlit as st
import torch
from PIL import Image
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

model = torch.load('plant-disease-model.pth', map_location=torch.device('cpu'))

# Define the image pre-processing function
def preprocessed_image(image):
    """Preprocess the image for classification."""

    image = Image.open(image)
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

# Define the main function
def main():
    """Classify a plant disease image using the trained model."""

    # Set the title and sidebar title
    st.title('Plant Disease Classification')
    st.sidebar.title('Plant Disease Classification App')

    # Display instructions and upload option
    st.markdown('Please upload an image of a plant leaf to classify the disease:')
    uploaded_file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])

    # If an image is uploaded, pre-process it and make predictions
    if uploaded_file is not None:
        image = preprocessed_image(uploaded_file)

        with torch.no_grad():
            output = model(torch.from_numpy(image)).squeeze()

        # Get the predicted class index
        predicted_class_index = np.argmax(output.numpy())

        # Get the predicted class name
        predicted_class_name = get_class_name(predicted_class_index)

        # Display the uploaded image and classification result
        st.image(uploaded_file, width=300)
        st.write('Predicted Class:', predicted_class_name)

        # Generate a PDF report
        generate_report(uploaded_file, predicted_class_name)

# Define the function to generate the PDF report
def generate_report(image_file, predicted_class_name):
    """Generate a PDF report with the uploaded image and predicted class name."""

    # Create a PDF object
    pdf = FPDF()
    pdf.add_page()

    # Add image to the PDF
    pdf.image(image_file, w=100, h=100)

    # Add text to the PDF
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(40, 10, 'Predicted Class:')
    pdf.set_font('Arial', '', 12)
    pdf.cell(60, 10, predicted_class_name)

    # Save the PDF report to the current directory
    report_name = 'plant_disease_report_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.pdf'
    pdf.output(report_name)

    # Show a message informing the user about the generated report
    st.write('A PDF report has been generated:')
    st.markdown('[' + report_name + '](./' + report_name + ')')

# Run the main function
if __name__ == '__main__':
    main()

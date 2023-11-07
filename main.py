# main.py
import streamlit as st
from PIL import Image
import torch
from fpdf import FPDF
import numpy as np

# Load the trained model
model = torch.load('/plant-disease-model.pth', map_location=torch.device('cuda'))
model.eval()

# Define the image pre-processing function
def preprocessed_image(image):
    image = Image.open(image)
    image = image.resize((224, 224), Image.ANTIALIAS)
    image = np.array(image)
    image = image.astype('float32') / 255
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

# Define the function to generate the PDF report
def generate_report(image_file, predicted_class_name):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Plant Disease Classification Report', 0, 1, 'C')

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Predicted Class: {predicted_class_name}', 0, 0, 'L')
            self.cell(0, 10, f'Image: {image_file.name}', 0, 0, 'C')
            self.cell(0, 10, f'Timestamp: {timestamp}', 0, 0, 'R')

    # Create a PDF object
    pdf = PDF()
    pdf.add_page()

    # Add image to the PDF
    pdf.image(image_file, x=10, y=20, w=100, h=100)

    # Add text to the PDF
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(40, 10, 'Predicted Class:', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, predicted_class_name, 0, 1, 'L')

    # Save the PDF report
    report_name = f'plant_disease_report_{timestamp}.pdf'
    pdf.output(report_name)

    return report_name

# Define the main function
def main():
    global timestamp
    timestamp = timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

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
        predicted_class_index = np.argmax(output.numpy())
        predicted_class_name = get_class_name(predicted_class_index)

        # Display the uploaded image and classification result
        st.image(uploaded_file, width=300)
        st.write('Predicted Class:', predicted_class_name)

        # Generate a PDF report
        report_name = generate_report(uploaded_file, predicted_class_name)

        # Show a message informing the user about the generated report
        st.write('A PDF report has been generated:')
        st.markdown(f'[{report_name}](./{report_name})')

# Run the main function
if __name__ == '__main__':
    main()

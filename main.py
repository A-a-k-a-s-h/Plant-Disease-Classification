import streamlit as st
from PIL import Image
import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from model import ResNet9

model = ResNet9(3, 38)  # num_classes is the number of classes in your dataset
model.load_state_dict(torch.load('plant-disease-model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define other necessary objects
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

classes = ['Raspberry___healthy', 'Tomato___Late_blight', 'Potato___healthy', 'Tomato___Leaf_Mold', 'Grape___healthy', 'Corn_(maize)___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 'Peach___healthy', 'Corn_(maize)___Common_rust_', 'Strawberry___Leaf_scorch', 'Grape___Black_rot', 'Squash___Powdery_mildew', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Apple___Cedar_apple_rust', 'Peach___Bacterial_spot', 'Tomato___healthy', 'Tomato___Target_Spot', 'Pepper,_bell___Bacterial_spot', 'Potato___Early_blight', 'Apple___healthy', 'Strawberry___healthy', 'Pepper,_bell___healthy', 'Tomato___Early_blight', 'Orange___Haunglongbing_(Citrus_greening)', 'Blueberry___healthy', 'Tomato___Tomato_mosaic_virus', 'Corn_(maize)___Northern_Leaf_Blight', 'Cherry_(including_sour)___healthy', 'Apple___Black_rot', 'Apple___Apple_scab', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Soybean___healthy', 'Tomato___Bacterial_spot', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Cherry_(including_sour)___Powdery_mildew', 'Grape___Esca_(Black_Measles)']


def predict_disease(uploaded_file, model, transform, classes):
    image = Image.open(uploaded_file).convert('RGB')
    image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        model.eval()
        output = model(image)

    _, predicted = torch.max(output, 1)
    prediction = classes[predicted.item()]

    return prediction

def main():
    st.title("Plant Disease Classification")
    st.write("Upload an image for disease prediction.")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Make prediction
        prediction = predict_disease(uploaded_file, model, transform, classes)

        # Display prediction
        st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()

import streamlit as st
from PIL import Image
import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from model import ResNet9

model = ResNet9(3, 38)  # num_classes is the number of classes in your dataset
model.load_state_dict(torch.load('plant-disease-model.pth', map_location=torch.device('cpu')), strict=False)
model.eval()  # Set the model to evaluation mode

# Define other necessary objects
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


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

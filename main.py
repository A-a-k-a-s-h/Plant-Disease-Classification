import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import models
from model import ConvBlock, ImageClassificationBase

# Load the model
class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x # ReLU can be applied before or after adding the input
        
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)  # out_dim: 128 x 64 x 64
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)  # out_dim: 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True)  # out_dim: 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))

    def forward(self, xb):  # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

model = ResNet9(3, 38)  # num_classes is the number of classes in your dataset
model.load_state_dict(torch.load('plant-disease-model.pth', map_location=torch.device('cpu')), strict=False)
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

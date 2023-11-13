# main.py

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

# Function to make predictions
def predict_disease(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return train.classes[predicted.item()]

# Model definition
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
        return self.relu2(out) + x  # ReLU can be applied before or after adding the input

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = SimpleResidualBlock()
        self.conv2 = SimpleResidualBlock()  # out_dim : 128 x 64 x 64
        self.res1 = nn.Sequential(SimpleResidualBlock(), SimpleResidualBlock())

        self.conv3 = SimpleResidualBlock()  # out_dim : 256 x 16 x 16
        self.conv4 = SimpleResidualBlock()  # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(SimpleResidualBlock(), SimpleResidualBlock())

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

# Load the trained model
model = ResNet9(3, num_diseases)  # Replace 'num_diseases' with the actual number of classes
model.load_state_dict(torch.load('plant-disease-model.pth'))  # Replace with the actual path

# Set the model to evaluation mode
model.eval()

# Function to make predictions
def predict_disease(image_path):
    image = Image.open(image_path).convert('RGB')
    image = ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return train.classes[predicted.item()]

# Streamlit app
def main():
    st.title("Plant Disease Classification")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        st.write("")
        st.write("Classifying...")

        # Make prediction
        prediction = predict_disease(uploaded_file)

        st.write(f"Prediction: {prediction}")

if __name__ == '__main__':
    main()

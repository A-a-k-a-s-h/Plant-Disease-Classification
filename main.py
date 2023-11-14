import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from model import ConvBlock, ImageClassificationBase, ResNet9

# Create a global model variable and load the model state dictionary
model = ResNet9(3, 38)
model.load_state_dict(torch.load('plant-disease-model.pth', map_location=torch.device('cuda')), strict=False)
model.eval()

# Define the predict_disease() function
def predict_disease(uploaded_file, model, transform, classes):
    # Set the model to evaluation mode
    model.eval()

    image = Image.open(uploaded_file).convert('RGB')
    image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image)

    _, predicted = torch.max(output, 1)
    prediction = classes[predicted.item()]

    return prediction

# Define the main() function
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

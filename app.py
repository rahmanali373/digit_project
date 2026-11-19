import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps

st.title("Handwritten Digit Prediction")
st.write("Upload a digit image (0–9)")

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class ViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(4, 32)
        self.position = nn.Parameter(torch.randn(1, 16, 32))
        layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = x.unfold(2, 2, 2).unfold(3, 2, 2)
        x = x.contiguous().view(x.size(0), 16, 4)
        x = self.embedding(x) + self.position
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))

cnn = CNNModel()
cnn.load_state_dict(torch.load("cnn_model.pth", map_location="cpu", weights_only=True))
cnn.eval()
vit = ViTModel()
vit.load_state_dict(torch.load("vit_model.pth", map_location="cpu", weights_only=True))
vit.eval()
uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Original Image")
    image = ImageOps.invert(image)
       image = ImageOps.invert(image)

    if np.array(image).std() < 5:
        st.error("Blank image! Please upload a digit.")
        st.stop()

    bbox = image.getbbox()

    if bbox is None:
        st.error("Blank image! Please upload a digit.")
        st.stop()

    image = image.crop(bbox)
    image.thumbnail((6, 6))

    canvas = Image.new("L", (8, 8), 0)
    canvas.paste(image, ((8-image.width)//2, (8-image.height)//2))

    image = canvas
    image_array = np.array(image) / 255.0
    st.image(
        image_array,
        caption="Processed 8x8 Image",
        width=200
    )

    input_tensor = torch.tensor(
        image_array, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        cnn_output = cnn(input_tensor)
        vit_output = vit(input_tensor)

        cnn_prediction = cnn_output.argmax(1).item()
        vit_prediction = vit_output.argmax(1).item()
    st.subheader("Prediction Results")
    st.write("CNN Prediction:", cnn_prediction)
    st.write("ViT Prediction:", vit_prediction)

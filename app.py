import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import tempfile
from scipy.signal import wiener

# ---------- CNN_Retino Class ----------
def findConv2dOutShape(Hin, Win, conv_layer):
    kernel_size = conv_layer.kernel_size[0]
    stride = conv_layer.stride[0]
    Hout = (Hin - kernel_size)//stride + 1
    Wout = (Win - kernel_size)//stride + 1
    return Hout, Wout

class CNN_Retino(nn.Module):
    def __init__(self, params):
        super(CNN_Retino, self).__init__()
        Cin,Hin,Win = params["shape_in"]
        init_f = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]

        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h,w=findConv2dOutShape(Hin,Win,self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv3)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv4)

        self.num_flatten = h*w*8*init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X,2,2)
        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)

# ---------- Deblur Function ----------
def deblur_image(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image uploaded.")
    
    # Wiener filter per channel, nan->0
    deblurred = np.zeros_like(img)
    for c in range(3):
        deblurred[:,:,c] = np.nan_to_num(wiener(img[:,:,c], (5,5)), nan=0.0)
    return deblurred.astype(np.uint8)

# ---------- Model Loading ----------
@st.cache_resource
def load_model(model_path, device):
    # Full model loading (weights + architecture)
    with torch.serialization.safe_globals([CNN_Retino]):
        model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model

# ---------- Transform ----------
transform = transforms.Compose([
    transforms.Resize((255,255)),  # match training input
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------- Prediction ----------
def predict_dr(model, img_pil, device):
    img = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs,1)
    return "Diabetic Retinopathy Detected" if predicted.item()==1 else "No Diabetic Retinopathy"

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="Diabetic Retinopathy Detector", layout="centered")
    st.title("🩺 Diabetic Retinopathy Detection")
    st.markdown("Upload a **blurred retina image**, it will be deblurred and checked for diabetic retinopathy.")

    uploaded_file = st.file_uploader("Upload Blurred Retina Image", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        st.subheader("Step 1️⃣: Image Deblurring")
        deblurred_img = deblur_image(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Original Blurred Image", use_container_width=True)
        with col2:
            st.image(deblurred_img, caption="Deblurred Image", use_container_width=True)

        # Save temporary deblurred file for PIL
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_file.name, deblurred_img)

        st.subheader("Step 2️⃣: Running Model Prediction...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "Retino_model.pt"  # your trained full-model checkpoint
        model = load_model(model_path, device)

        img_pil = Image.open(temp_file.name).convert("RGB")
        result = predict_dr(model, img_pil, device)

        st.success(f"✅ Result: **{result}**")

if __name__=="__main__":
    main()

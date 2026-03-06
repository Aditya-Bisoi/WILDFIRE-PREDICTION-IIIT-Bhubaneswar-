import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import pydeck as pdk
import os
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. MODEL ARCHITECTURE (Matches your code) ---
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --- 2. CONFIG & UTILS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@st.cache_resource
def load_model():
    model = CNNModel().to(device)
    # model.load_state_dict(torch.load("your_model.pth", map_location=device))
    model.eval()
    return model

def parse_filename(filename):
    """Extracts coordinates from your specific filename format: 'long,lat.jpg'"""
    try:
        clean_name = filename.replace('.jpg', '').replace('.png', '')
        parts = clean_name.split(',')
        return float(parts[0]), float(parts[1])
    except:
        return None, None

# --- 3. UI DASHBOARD ---
st.set_page_config(page_title="Wildfire Monitor", layout="wide", page_icon="🔥")
st.title("🛰️ Predict Wildfire Monitor")
st.sidebar.header("Control Panel")

# Load model
model = load_model()

# File Uploader
uploaded_files = st.sidebar.file_uploader("Upload Image(s)", type=['jpg', 'png'], accept_multiple_files=True)

if uploaded_files:
    map_data = []
    
    for uploaded_file in uploaded_files:
        # Prediction Logic
        img = Image.open(uploaded_file).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            score, pred = torch.max(prob, 1)
        
        # Metadata logic
        lon, lat = parse_filename(uploaded_file.name)
        status = "RISK" if pred.item() == 1 else "NO RISK"
        
        # FIX: Define the color here based on the status
        point_color = [255, 69, 0, 160] if status == "RISK" else [0, 255, 0, 160]
        
        if lon and lat:
            map_data.append({
                "lon": lon, 
                "lat": lat, 
                "status": status, 
                "confidence": score.item(),
                "color": point_color  # Added color to the dictionary
            })

    # --- TOP METRICS ---
    risk_count = sum(1 for d in map_data if d['status'] == "RISK")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Images", len(uploaded_files))
    m2.metric("Wildfire Risk Detected", risk_count, delta_color="inverse")
    m3.metric("System Status", "Live/Monitoring", delta="Active")

    # --- MAIN CONTENT TABS ---
    tab1, tab2 = st.tabs(["📍 Geospatial Monitor", "🔍 AI Diagnostics (Grad-CAM)"])

    with tab1:
        st.subheader("Regional Risk Distribution")
        if map_data:
            df_map = pd.DataFrame(map_data)
            
            # FIX: Point get_color directly to the "color" column
            layer = pdk.Layer(
                "ScatterplotLayer",
                df_map,
                get_position=["lon", "lat"],
                get_color="color",  
                get_radius=5000,
                pickable=True
            )
            
            # Center map on the data points
            avg_lat = df_map['lat'].mean()
            avg_lon = df_map['lon'].mean()
            
            st.pydeck_chart(pdk.Deck(
                layers=[layer], 
                initial_view_state=pdk.ViewState(
                    latitude=avg_lat, 
                    longitude=avg_lon, 
                    zoom=5, 
                    pitch=45
                )
            ))
        else:
            st.info("Upload images with filenames like 'longitude,latitude.jpg' to see them on the map.")
    with tab2:
        st.subheader("Deep Learning Interpretability")
        selected_file = st.selectbox("Select image for XAI analysis", [f.name for f in uploaded_files])
        
        # Process single image for Grad-CAM
        curr_file = next(f for f in uploaded_files if f.name == selected_file)
        curr_img = Image.open(curr_file).convert('RGB')
        curr_tensor = transform(curr_img).unsqueeze(0).to(device)
        
        # Grad-CAM visualization
        target_layers = [model.conv_layers[-3]]
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=curr_tensor)[0, :]
        rgb_img = np.array(curr_img.resize((128, 128))) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        c1, c2 = st.columns(2)
        c1.image(curr_img, caption="Input Image", use_column_width=True)
        c2.image(cam_image, caption="AI Focus Area (Grad-CAM)", use_column_width=True)

else: 
    st.warning("Please upload forest imagery in the sidebar to begin monitoring.")
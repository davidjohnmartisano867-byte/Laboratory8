import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="YOLO Fruit Detection", layout="wide", page_icon="🍎")

# --- TITLE ---
st.title("🍎 YOLOv8 Fruit Detection App")
st.markdown("Upload an image of a fruit to detect and classify it using the trained YOLO model.")

# --- LOAD MODEL (Cached para hindi mabagal) ---
@st.cache_resource
def load_model():
    model_path = "weights/best.pt"
    return YOLO(model_path)

try:
    with st.spinner("Loading YOLO Model..."):
        model = load_model()
    st.success("Model loaded successfully!", icon="✅")
except Exception as e:
    st.error(f"Error loading model: Make sure 'weights/best.pt' exists. ({e})")
    st.stop()

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # I-convert ang uploaded file para mabasa ng OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    if image is not None:
        # Gumawa ng dalawang kolumn para sa comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # Mag-predict gamit ang model
        with st.spinner('🔍 Detecting fruits...'):
            results = model.predict(source=image, conf=0.25)
            
        # Kunin ang image na may nakadrawing na bounding boxes
        annotated_img = results[0].plot()
        
        with col2:
            st.subheader("🎯 Detection Result")
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # --- I-display ang mga nakitang details ---
        st.divider()
        st.subheader("📋 Detection Details")
        
        boxes = results[0].boxes
        if len(boxes) == 0:
            st.warning("No fruits detected.")
        else:
            # I-convert sa dataframe para maganda tingnan
            detected_data = []
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                conf = float(box.conf[0])
                detected_data.append({"Fruit Detected": cls_name.title(), "Confidence Score": f"{conf:.2%}"})
            
            st.dataframe(detected_data, use_container_width=True, hide_index=True)
            st.success(f"Successfully detected {len(boxes)} fruit(s)!", icon="🎉")
    else:
        st.error("Cannot read the image. Please upload a valid image file.")
else:
    st.info("👆 Please upload an image file to get started.")
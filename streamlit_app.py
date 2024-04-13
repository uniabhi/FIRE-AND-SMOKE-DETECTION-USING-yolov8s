import PIL
import numpy as np
import streamlit as st
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd


# good prediction model - models/yolov8/weights/best (1).pt and latest yolov8 on custom dataset 
#  latest yolov8 on custom dataset = 'models/best (7).pt'
model_yolov8 = 'weights/best.pt'

st.set_page_config(
    page_title="Forest Fire and Smoke Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.header("Image Config")
    uploaded_file = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", "bmp", "webp")
    )
    confidence = float(st.slider("Select Model Confidence", 15, 20, 100)) / 100

st.title("Fire and Smoke Detection using YOLOv8")
st.caption("Using custom dataset")

st.write("Model Characteristics:")
chars_alt = pd.DataFrame({
    "Param": ["Dataset", "Images", "Epochs", "IMG_SIZE", "BATCH_SIZE", "LR"],
    "Value": ["https://universe.roboflow.com/ds/Ng1WjvLh9i?key=JBrja8Xhvb", "3974", "50", "640", "16", "0.01"]
})
st.table(chars_alt)

st.caption(
    'Upload a photo and then click the "Detect Objects" button to view the results.'
)

col1, col2 = st.columns(2)

with col1:
    if uploaded_file:
        if uploaded_file.type in ["image/jpeg", "image/png", "image/bmp", "image/webp"]:
            uploaded_image = PIL.Image.open(uploaded_file)
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

@st.cache_resource
def load_yolov8_model():
    return YOLO(model_yolov8)

# Load YOLOv8 model
try:
    model = load_yolov8_model()
except Exception as ex:
    st.error(f"Unable to load YOLOv8 model. Check the specified path: {model_yolov8}")
    st.error(ex)

def process_image_detections(res, col2):
    if res[0].boxes.shape[0] == 0:
        col2.write("No fire or smoke detected.")
        return
    
    print(res[0].speed)
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    col2.image(res_plotted, caption="Detected Image", use_column_width=True)

    fire_count = 0
    smoke_count = 0
    fire_confidences = []
    smoke_confidences = []

    with st.expander("Detection Results"):
        for box in boxes:
            label = model.names[int(box.cls)]
            confidence = box.conf.item()
            if label == "fire":
                fire_count += 1
                fire_confidences.append(confidence)
                st.markdown(f"<span style='color:red'>{label.capitalize()}</span> - Confidence: {confidence:.2f} - Coordinates: {box.xywh}", unsafe_allow_html=True)
            elif label == "smoke":
                smoke_count += 1
                smoke_confidences.append(confidence)
                st.markdown(f"<span style='color:blue'>{label.capitalize()}</span> - Confidence: {confidence:.2f} - Coordinates: {box.xywh}", unsafe_allow_html=True)
        st.write(f"Total Fires Detected: {fire_count}")
        st.write(f"Total Smokes Detected: {smoke_count}")
        st.markdown(f"<span style='color:green'>Preprocess Time:</span> {res[0].speed.get('preprocess'):.2f} ms<br><span style='color:green'>Inference Time:</span> {res[0].speed.get('inference'):.2f} ms<br><span style='color:green'>Postprocess Time:</span> {res[0].speed.get('postprocess'):.2f} ms", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Average Confidence per Class"):
            labels = ["Fire", "Smoke"]
            avg_confidences = [np.mean(fire_confidences) if fire_confidences else 0, 
                            np.mean(smoke_confidences) if smoke_confidences else 0]
            plt.figure(figsize=(8, 5))
            plt.bar(labels, avg_confidences, color=['#ff9999','#66b2ff'])
            plt.title("Average Confidence per Class")
            plt.ylabel("Average Confidence")
            plt.ylim(0, 1)  # Y-axis between 0 and 1 for clarity
            col1.pyplot(plt)

    with col2:
        with st.expander("Class Distribution"):
            labels = ["Fire", "Smoke"]
            sizes = [fire_count, smoke_count]
            colors = ['#ff9999','#66b2ff']
            if any(sizes):  # Check if sizes list is not empty
                plt.figure(figsize=(8, 8))
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                plt.title("Distribution of Detected Classes")
                col2.pyplot(plt)
            else:
                col2.write("No fire or smoke detected.")



if st.sidebar.button("Detect Objects"):
    if uploaded_file.type in ["image/jpeg", "image/png", "image/bmp", "image/webp"]:
        # Perform detection on the uploaded image
        res = model.predict(uploaded_image, conf=confidence)
        process_image_detections(res, col2)

# problem - fire and smoke count = 0 and overlapping bounding boxes 0bject
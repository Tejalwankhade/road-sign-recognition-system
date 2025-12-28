import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd

# Load model
model = load_model("traffic_sign_cnn.h5")

# Load label names
labels = pd.read_csv("Train.csv")["ClassId"].unique()

st.title("Traffic Sign Recognition App")

uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(64,64))
    st.image(img, caption="Uploaded image")

    x = image.img_to_array(img)
    x = x/255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    class_index = pred.argmax()

    st.success(f"Predicted Traffic Sign Class: {class_index}")

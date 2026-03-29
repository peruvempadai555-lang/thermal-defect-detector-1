import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Layer, Conv2D, concatenate

# ---------- CUSTOM LAYER DEFINITION ----------
class fire_module(Layer):
    def __init__(self, squeeze, expand, **kwargs):
        super().__init__(**kwargs)
        self.squeeze = squeeze
        self.expand = expand
        self.conv_squeeze = Conv2D(squeeze, (1,1), activation='tanh', padding='same')
        self.conv_expand1 = Conv2D(expand, (1,1), activation='tanh', padding='same')
        self.conv_expand3 = Conv2D(expand, (3,3), activation='tanh', padding='same')
    def call(self, inputs):
        s = self.conv_squeeze(inputs)
        e1 = self.conv_expand1(s)
        e3 = self.conv_expand3(s)
        return concatenate([e1, e3])
    def get_config(self):
        config = super().get_config()
        config.update({'squeeze': self.squeeze, 'expand': self.expand})
        return config

# ---------- LOAD MODEL WITH CUSTOM OBJECT ----------
import requests

url = "https://huggingface.co/spaces/M-Parames01/thermal-defect-model/resolve/main/fusion_resume_model_1.keras?download=true"
response = requests.get(url)
with open("fusion_resume_model_1.keras", "wb") as f:
    f.write(response.content)

# Model load
model = tf.keras.models.load_model("fusion_resume_model_1.keras", custom_objects={'fire_module': fire_module}, compile=False)

# ---------- CLASS NAMES ----------
CLASS_NAMES = ['No_Defect', 'Minor_Defect', 'Major_Defect']

# ---------- PAGE UI ----------
st.title("🔥 Thermal Image Defect Detection")
uploaded_file = st.file_uploader("Choose a thermal image...", type=["jpg","jpeg","png"])

if uploaded_file:
    # Preprocess
    img = Image.open(uploaded_file).convert('RGB').resize((224,224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    pred = model.predict(img_array)
    pred_class = CLASS_NAMES[np.argmax(pred[0])]
    confidence = np.max(pred[0])
    
    # Display
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    st.success(f"Prediction: {pred_class}")
    st.metric("Confidence", f"{confidence:.2%}")
    


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "asl_sign_classifier.keras"

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

st.title("ASL Sign Language Classifier")
st.write("Upload an image of an ASL hand sign to predict its class.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=300)

    if st.button("Predict Sign"):
        model = load_model()
        resized_image = image.resize((64, 64))
        image_array = np.array(resized_image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))

        st.success(f"Predicted sign: {CLASS_NAMES[predicted_index]}")
        confidence = float(predictions[0][predicted_index]) * 100
        st.write(f"Confidence: {confidence:.2f}%")

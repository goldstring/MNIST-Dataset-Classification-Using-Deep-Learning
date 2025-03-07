import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the model once when the app starts
@st.cache_resource
def load_mnist_model():
    model = load_model('mnist_classification_deep_learning_model.h5')
    return model

# Prediction function
def predict_image(model, image, isColorImage):
    try:
        # Read the image file as bytes
        image_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if isColorImage:
            # Process color image: turn white to black and convert to grayscale
            lower_white = np.array([200, 200, 200])
            upper_white = np.array([255, 255, 255])
            white_mask = cv2.inRange(img, lower_white, upper_white)
            img[white_mask != 0] = [0, 0, 0]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize the image to 28x28
        resized_img = cv2.resize(img, (28, 28))

        # Normalize the image
        img_array = resized_img / 255.0

        # Reshape the image to match the model input (batch size, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = prediction.argmax()

        # Display the result
        st.image(resized_img, caption='Processed Image (28x28 Grayscale)', channels='GRAY')
        st.write(f"### Predicted Class: {predicted_class}")

        return predicted_class

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Main function to run the Streamlit app
def main():
    st.title("MNIST Digit Classification")

    # Load the model
    model = load_mnist_model()

    # Upload the image
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    # Checkbox to determine if the image is a color image
    isColorImage = st.checkbox("Is the image a color image?", value=False)

    if uploaded_file is not None:
        st.write("### Uploaded Image")
        
        st.image(uploaded_file, caption='Uploaded Image',width=50)

        # Make a prediction button
        if st.button("Predict"):
            predict_image(model, uploaded_file, isColorImage)

# Run the app
if __name__ == "__main__":
    main()

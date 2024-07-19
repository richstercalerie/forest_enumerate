import cv2
import numpy as np
from PIL import Image
import streamlit as st
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def preprocess_image(image):
    image_array = np.array(image)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (11, 11), 0)
    _, binary_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY_INV)
    return binary_image, image_array

def find_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def count_trees(image):
    binary_image, image_array = preprocess_image(image)
    contours = find_contours(binary_image)
    num_trees = len(contours)
    contoured_image = image_array.copy()
    cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 3)
    return num_trees, contoured_image

def main():
    # Add the background image
    add_bg_from_local("count.png")
    
    st.markdown("<h1 style='text-align: center; color: brown;'>Van-Ganaka: A Tree Enumerator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: brown;'>Upload an image to count the number of trees.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        num_trees, contoured_image = count_trees(image)
        
        st.image(contoured_image, caption=f"Processed Image with {num_trees} Trees.", use_column_width=True)
        st.markdown(f"<h3 style='text-align: center; color: brown;'>Number of trees detected: {num_trees}</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

import os
import streamlit as st # type: ignore
import numpy as np
import pickle
import tensorflow as tf
from keras.layers import GlobalMaxPooling2D # type: ignore
from keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from PIL import Image # type: ignore

# Load feature vectors and filenames
feature_list = np.array(pickle.load(open('featurevector.pkl', 'rb')))
filenames = pickle.load(open('filename.pkl', 'rb'))

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D(),
    tf.keras.layers.Dense(512, activation='relu')
])

# Create uploads directory if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

# Function to extract features from an image using the model
def extract_feature(img_path, model):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array).flatten()
        normalized = features / norm(features)
        return normalized
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Function to find nearest neighbors and recommend images
def recommend(features, feature_list):
    try:
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices
    except Exception as e:
        st.error(f"Error finding recommendations: {e}")
        return None

# Website configuration
st.set_page_config(page_title="Fashion Recommendation System", page_icon=":tada:", layout="wide")

# Background image styling
page_bg_img = """
<style>
body {
    background-image: url("https://plus.unsplash.com/premium_photo-1714226985393-cefb821470c7?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
}
</style>
"""

# Apply the background image
st.markdown(page_bg_img, unsafe_allow_html=True)

# Trending Images section
st.title("Trending Now")
img_col_1 = Image.open("t1.jpg")
img_col_2 = Image.open("t2.jpg")
img_col_3 = Image.open("t3.jpg")
img_col_4 = Image.open("t4.jpg")
img_col_5 = Image.open("t5.jpg")
img_col_6 = Image.open("t6.jpeg")

with st.container():
    col_1, col_2, col_3, col_4, col_5, col_6 = st.columns(6)
    with col_1:
        st.image(img_col_1, use_column_width=True)
    with col_2:
        st.image(img_col_2, use_column_width=True)
    with col_3:
        st.image(img_col_3, use_column_width=True)
    with col_4:
        st.image(img_col_4, use_column_width=True)
    with col_5:
        st.image(img_col_5, use_column_width=True)
    with col_6:
        st.image(img_col_6, use_column_width=True)

# File Upload and Recommendation section
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        resized_img = display_image.resize((200, 200))
        st.image(resized_img, caption="Uploaded Image")

        # Extract features from the uploaded image
        features = extract_feature(os.path.join("uploads", uploaded_file.name), model)
        if features is not None and features.shape[0] == 512:
            st.title("Recommendations")
            # Get recommendations
            indices = recommend(features, feature_list)
            if indices is not None:
                # Display recommendations
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.image(filenames[indices[0][0]], use_column_width=True)
                with col2:
                    st.image(filenames[indices[0][1]], use_column_width=True)
                with col3:
                    st.image(filenames[indices[0][2]], use_column_width=True)
                with col4:
                    st.image(filenames[indices[0][3]], use_column_width=True)
                with col5:
                    st.image(filenames[indices[0][4]], use_column_width=True)
        else:
            st.error("Feature extraction error or feature shape mismatch. Please try again.")
    else:
        st.error("Failed to save the uploaded file.")
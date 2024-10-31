import os
import io

import requests
from dotenv import load_dotenv
from PIL import Image

import streamlit as st


# Load the environment variables
folder_path = os.path.dirname(__file__)
load_dotenv(os.path.join(folder_path, ".env"))

# Set which host to skip proxy
host = os.getenv("NO_PROXY_HOST")
os.environ["NO_PROXY"] = host

# URL of the server endpoint
API_URL = f"http://{host}/search-image"

# Title of the application
st.title("Fashion Image Search Engine 🔎")

# Create chatbot interface to input text query or upload image
option = st.selectbox("Select search option", ["Text", "Image"])

# Number of results
# top_k = st.number_input("Number of results", min_value=1, value=3)
top_k = 3

def display_images_in_rows(image_urls, images_per_row=3):
    for i in range(0, len(image_urls), images_per_row):
        cols = st.columns(images_per_row)
        for j, col in enumerate(cols):
            if i + j < len(image_urls):
                col.image(image_urls[i + j])

if option == "Text":
    # Input text query
    query = st.text_input("Enter text query")
    search_button = st.button("Search")

    if search_button or query:
        if not query:
            st.error("Please enter a text query")
            st.stop()

        # Send the text query to the server
        params = {"query": query, "top_k": top_k}
        response = requests.post(API_URL, params=params, timeout=600)
        if response.status_code == 200:
            # Display the search results
            results = response.json()
            st.write("Search results:")
            display_images_in_rows(results["urls"])
        else:
            st.error(f"Error processing the request ({response.status_code})")
elif option == "Image":
    # Upload image
    image_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        # Display the uploaded image
        image = Image.open(io.BytesIO(image_file.read()))
        st.image(image, width=100)
    if st.button("Search"):
        if image_file is None:
            st.error("Please upload an image")
            st.stop()
        # Send the image to the server
        params = {"top_k": top_k}
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
        response = requests.post(API_URL, files=files, params=params, timeout=600)
        if response.status_code == 200:
            # Display the search results
            results = response.json()
            st.write("Search results:")
            display_images_in_rows(results["urls"])
        else:
            st.error("Error processing the request")

# Display the footer
st.markdown("---")
st.write("Built in Oct 2024 by Tai Le 👨‍💻")
# app/ml/clip_inference.py

import requests
import os

# Replace with your Hugging Face API token
HUGGING_FACE_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")  # Make sure to set this in your .env file
HEADERS = {
    "Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}"
}

# URL of the CLIP model on Hugging Face
CLIP_MODEL_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"

def get_image_and_text_embeddings(image_path, text_query):
    """Send the image and text query to the Hugging Face API and get embeddings."""
    with open(image_path, "rb") as image_file:
        files = {
            "image": image_file,
            "text": text_query
        }
        response = requests.post(CLIP_MODEL_URL, headers=HEADERS, files=files)

    if response.status_code == 200:
        return response.json()  # Process as needed
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

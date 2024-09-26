# app/ml/process_image.py

import io
from PIL import Image
import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def process_image(image_bytes):
    """Process the uploaded image and return its embedding."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings.numpy().tolist()  # Convert to list for Milvus

def get_text_embedding(query):
    """Convert text to embedding."""
    inputs = processor(text=query, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)
    return embeddings.numpy().tolist()  # Convert to list for Milvus

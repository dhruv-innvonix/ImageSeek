# app/ml/process_text.py
import numpy as np
from transformers import CLIPModel, CLIPProcessor

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

async def process_text(query: str):
    # Process the text input
    inputs = processor(text=query, return_tensors="pt", padding=True)

    # Generate embedding
    outputs = model.get_text_features(**inputs)
    embedding = outputs.detach().numpy()[0]  # Convert tensor to numpy array

    # Normalize the embedding
    normalized_embedding = embedding / np.linalg.norm(embedding)

    return normalized_embedding.tolist()

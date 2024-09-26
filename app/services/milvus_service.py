# app/services/milvus_service.py

from pymilvus import (
    Collection,
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
)
from app.utils.milvus_utils import create_collection_if_not_exists, get_collection_schema
from app.schemas.image import ImageSchema  # Assuming this schema exists
from app.ml.process_image import process_image, get_text_embedding  # Function to process images and convert text to embeddings
from pathlib import Path
import os
from fastapi import UploadFile
from datetime import datetime
from PIL import Image

# Define connection parameters
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "images"

UPLOAD_FOLDER = "static/uploads"
class MilvusService:
    UPLOAD_FOLDER = "static/uploads"
    def __init__(self):
        # Connect to Milvus
        self.connect_to_milvus()

        # Create collection if it doesn't exist
        create_collection_if_not_exists(COLLECTION_NAME)

    def connect_to_milvus(self):
        """Connect to the Milvus instance."""
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    def save_image(self, image: UploadFile, image_id: str) -> str:
        """Save the uploaded image locally and return the file path."""
        # Generate a timestamp string for the filename
        timestamp = datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
        image_filename = f"{timestamp}.png"  # Create filename based on timestamp

        # Construct the full file path
        image_path = os.path.join(self.UPLOAD_FOLDER, image_filename)

        # Ensure the upload folder exists
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)

        try:
            # Open the image using PIL to validate it
            with Image.open(image.file) as img:
                img = img.convert("RGB")  # Convert image to RGB if necessary
                img.save(image_path, format='PNG')  # Save the image as PNG

            return image_path
        except Exception as e:
            print(f"Error saving image: {e}")
            raise None

    def insert_image_embedding(self, image_id: str, embedding: list, image: UploadFile, category: str = "default"):
        """Insert an image embedding into the Milvus collection and store the image locally."""
        collection = Collection(COLLECTION_NAME)

        # Flatten embedding if nested
        if isinstance(embedding[0], list):
            embedding = embedding[0]

        # Save the image locally
        image_path = self.save_image(image, image_id)

        # Insert embedding data along with image path and category
        data = [[image_id], [embedding], [category], [image_path]]  # Storing image_path along with embedding

        if not collection.has_index():
            print("Index not found. Creating index on embedding field...")
            index_params = {
                "index_type": "IVF_FLAT",  # Index type, you can choose others like IVF_SQ8, HNSW, etc.
                "metric_type": "L2",       # Metric type, for float vectors, L2 (Euclidean distance) is common
                "params": {"nlist": 128}   # Index parameter, nlist can be tuned for performance
            }
            collection.create_index(field_name="embedding", index_params=index_params)
        
        # Load the collection if needed
        collection.load()

        # Insert data into Milvus
        collection.insert(data)
        print(f"Inserted image {image_id} into Milvus with the file at {image_path}.")

    def search_images(self, query: str, top_k: int = 5):
        """Search for similar images based on a text query and return the file paths."""
        # Convert the query text to an embedding
        query_embedding = get_text_embedding(query)
        
        # Ensure query_embedding is a flat list of floats
        if isinstance(query_embedding[0], list):
            query_embedding = query_embedding[0]
        
        collection = Collection(COLLECTION_NAME)

        # Load the collection into memory if not already loaded
        if collection.is_empty:
            collection.load()

        # Perform search (data should be a list of lists: [[embedding]])
        results = collection.search(
            data=[query_embedding],  # Milvus expects the embedding to be a list of lists
            anns_field="embedding",  # The field in Milvus where embeddings are stored
            param={"nprobe": 10},    # Search parameter, can be tuned for performance/accuracy
            limit=top_k              # Number of results to return
        )

        # Extract the image paths from the search results
        image_paths = []
        for hit in results[0]:
            # Extract the 'id' part from the string (assuming it follows the format you provided)
            hit_str = str(hit)  # Convert hit to string
            print("hit_str:", hit_str)

            # Extract the image ID (assuming the format "id: <image_id>, distance: ...")
            id_part = hit_str.split(",")[0]  # Extract up to the first comma
            image_id = id_part.split(":")[1].strip()  # Extract the actual image ID and remove spaces
            
            # Reconstruct the image path (assuming you store images with their ID as filename)
            image_path = os.path.join(UPLOAD_FOLDER, image_id)
            
            # Check if the image file exists before adding it to the results
            if os.path.exists(image_path):
                image_paths.append(image_path)
            else:
                print(f"Image {image_id} not found at {image_path}.")

        # Return the file paths (to be used by FastAPI to send back the images)
        return image_paths

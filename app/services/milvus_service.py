# app/services/milvus_service.py

from pymilvus import (
    Collection,
    connections,
)
from app.utils.milvus_utils import create_collection_if_not_exists
from app.ml.process_image import get_text_embedding  # Function to process images and convert text to embeddings
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
        self.collection = Collection(COLLECTION_NAME)

    def connect_to_milvus(self):
        """Connect to the Milvus instance."""
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    def save_image(self, image: UploadFile, image_id: str) -> str:
        """Save the uploaded image locally and return the file path."""
        # Generate a timestamp string for the filename
        image_filename = f"{image_id}.png"  # Create filename based on timestamp

        # Construct the full file path
        image_path = os.path.join(self.UPLOAD_FOLDER, image_filename)

        # Ensure the upload folder exists
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)

        try:
            # Open the image using PIL to validate it
            with Image.open(image.file) as img:
                img = img.convert("RGB")  # Convert image to RGB if necessary
                img.save(image_path, format='PNG')  # Save the image as PNG

            return image_filename
        except Exception as e:
            print(f"Error saving image: {e}")
            raise None

    def insert_image_embedding(self, 
                                image: UploadFile,
                                image_id: str,
                                embedding: list, 
                                category: str = "uncategorized",
                                title: str = "Untitled",
                                description: str = "No description provided.",
                                tags: str = "",  # Empty string indicates no tags
                                location: str = "Unknown",
                                date_taken: str = "Not specified",
                                uploader: str = "anonymous",  # Default uploader name
                                quality_rating: float = 0.0  # Assuming 0 is the lowest quality rating
                                ):
        """Insert an image embedding into the Milvus collection and store the image locally."""
        collection = Collection(COLLECTION_NAME)

        # Flatten embedding if nested
        if isinstance(embedding[0], list):
            embedding = embedding[0]

        # Save the image locally
        image_filename = self.save_image(image, image_id)

        # Insert embedding data along with image path and category
        data = [
            [image_filename],      # image_id
            [embedding],           # embedding
            [category],            # category
            [title],               # title (optional)
            [description],         # description (optional)
            [tags],                # tags (optional)
            [location],            # location (optional)
            [date_taken],          # date_taken (optional)
            [uploader],            # uploader (optional)
            [float(quality_rating)]       # quality_rating (optional)
        ]
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
        print(f"Inserted image {image_id} into Milvus with the file at {image_filename}.")

    def search_images(self, query: str, category:str, top_k: int = 10,):
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
            limit=top_k,              # Number of results to return
            output_fields=["image_id", "category", "title", "description", "tags", "location", "date_taken", "uploader", "quality_rating"]  # Specify the fields to return
        )

        # Extract the image paths from the search results
        image_paths = []
        for hit in results[0]:
            # if hit.score > 160:
            #     continue

            image_id = hit.id
            image_path = os.path.join(UPLOAD_FOLDER, image_id)
            
            # Check if the image file exists before adding it to the results
            if os.path.exists(image_path):
                # TODO: filter the image based on metadata and remove non-relevent
                if hit.fields['category'] and hit.fields['category'] != category:
                    continue
                image_paths.append(image_path)
                
            else:
                print(f"Image {image_id} not found at {image_path}.")
       
        return image_paths

    def get_image_by_embedding(self, embedding, top_k=10):
        # Query Milvus to check if an image with the same embedding exists
        # Note: You may need to adjust this according to your Milvus setup
        results = self.collection.search(
            data=embedding,  # Milvus expects the embedding to be a list of lists
            anns_field="embedding",  # The field in Milvus where embeddings are stored
            param={"nprobe": 10},    # Search parameter, can be tuned for performance/accuracy
            limit=top_k,              # Number of results to return
            # output_fields=["image_id", "category", "title", "description", "tags", "location", "date_taken", "uploader", "quality_rating"]  # Specify the fields to return
        )
        
        # If the results contain any hits, return the first one
        image_paths = []
        for hit in results[0]:
            if hit.score < 150:
                image_path = os.path.join(UPLOAD_FOLDER, hit.id)
                if os.path.exists(image_path):
                    image_paths.append(image_path)
        return image_paths
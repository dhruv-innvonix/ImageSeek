# app/api/v1/image_upload.py

from fastapi import APIRouter, UploadFile, File
from app.services.milvus_service import MilvusService
from app.ml.process_image import process_image  # Import the image processing function

router = APIRouter()
milvus_service = MilvusService()

@router.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image and store its embedding in Milvus."""
    image = file
    image_id = file.filename  # Use filename or generate a unique ID
    image_bytes = await file.read()  # Read the image bytes
    embedding = process_image(image_bytes)  # Process the image and get embedding
    category = "default"  # You can modify this to categorize images as needed

    milvus_service.insert_image_embedding(image_id, embedding, image, category)
    return {"message": "Image uploaded successfully", "image_id": image_id}

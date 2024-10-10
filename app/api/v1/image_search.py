# app/api/v1/image_search.py

import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from app.services.milvus_service import MilvusService
from app.ml.process_image import process_image 

router = APIRouter()
milvus_service = MilvusService()


@router.get("/search/")
async def search_images(query: str, category:str = None, file: UploadFile = File(...)):
    """Search for images and return the image files."""
    if file:
        image_bytes = await file.read()  # Read the image bytes
        embedding = process_image(image_bytes)  # Process the image and get embedding
        image_paths = milvus_service.get_image_by_embedding(embedding)
        # image_paths = [obj.id for obj in objs]
    else:    
        image_paths = milvus_service.search_images(query, category)

    # Return the actual image files
    return [FileResponse(path) for path in image_paths if os.path.exists(path)]

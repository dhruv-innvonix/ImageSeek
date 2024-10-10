# app/api/v1/image_upload.py

from fastapi import APIRouter, UploadFile, File, Form
from app.services.milvus_service import MilvusService
from app.ml.process_image import process_image  # Import the image processing function
from typing import Annotated
import uuid
from datetime import datetime, date

router = APIRouter()
milvus_service = MilvusService()

@router.post("/upload/")
async def upload_image(category: Annotated[str, Form()] = "uncategorized",
                        title: Annotated[str, Form()] = "Untitled",
                        description: Annotated[str, Form()] = "No description provided.",
                        tags: Annotated[str, Form()] = "",
                        uploader: Annotated[str, Form()] = "anonymous",
                        file: UploadFile = File(...)
                    ):
    """Upload an image and store its embedding in Milvus."""

    image_bytes = await file.read()  # Read the image bytes
    embedding = process_image(image_bytes)  # Process the image and get embedding


    milvus_service.insert_image_embedding(
        image=file,
        image_id=uuid.uuid4(),
        embedding=embedding,
        category=category,
        title=title,
        description=description,
        tags=tags,
        location="India",
        date_taken=datetime.now().date().isoformat(),
        uploader=uploader,
        quality_rating=5
    )
    return {"message": "Image uploaded successfully"}

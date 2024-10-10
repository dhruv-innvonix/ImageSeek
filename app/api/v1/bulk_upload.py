# app/api/v1/image_upload.py

from fastapi import APIRouter
from app.services.milvus_service import MilvusService
from app.ml.process_image import process_image  # Import the image processing function
from fastapi import APIRouter, HTTPException
import uuid
from datetime import datetime
import os
from pathlib import Path
from io import BytesIO

router = APIRouter()
milvus_service = MilvusService()

STATIC_FOLDER_PATH = Path("static/fruit") 

@router.get("/bulk-upload/")
async def bulk_upload():

    if not STATIC_FOLDER_PATH.exists():
        raise HTTPException(status_code=404, detail="Static folder not found")

    image_files = [f for f in os.listdir(STATIC_FOLDER_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    if not image_files:
        raise HTTPException(status_code=404, detail="No images found in the static folder")
    
    upload_results = []
    for image_file in image_files:
        image_path = STATIC_FOLDER_PATH / image_file
        try:
            with open(image_path, "rb") as file:
                image_bytes = file.read()  # Read the image bytes
                embedding = process_image(image_bytes)  # Process the image and get embedding
                
                # Check if the image already exists in Milvus
                # existing_image = milvus_service.get_image_by_embedding(embedding)
                # if existing_image:
                #     print(f"{image_file} already exists in Milvus, skipping upload.")
                #     upload_results.append({"file": image_file, "status": "skipped", "reason": "already exists"})
                #     continue
                
                # Create a BytesIO object to mimic an UploadFile
                # file_like = BytesIO(image_bytes)
                # file_like.name = image_file  # Assign the name attribute

                milvus_service.insert_image_embedding(
                    image=file,
                    image_id=uuid.uuid4(),
                    embedding=embedding,
                    category="fruit",
                    title="fruit",
                    # description=description,
                    tags="fruit",
                    location="India",
                    date_taken=datetime.now().date().isoformat(),
                    # uploader=uploader,
                    quality_rating=5
                )
                print(f"{image_file} uploaded successfully")
                upload_results.append({"file": image_file, "status": "success"})
                break
        except Exception as e:
            print(f"Error uploading {image_file}: {e}")
            upload_results.append({"file": image_file, "status": "error", "detail": str(e)})

    return {"message": "Image uploaded successfully"}

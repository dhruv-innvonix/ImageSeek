# app/api/v1/image_search.py

from fastapi import APIRouter
from app.services.milvus_service import MilvusService

router = APIRouter()
milvus_service = MilvusService()

# @router.get("/search/")
# async def search_images(query: str, threshold: float = 200.0):
#     """Search for images based on a text query, with a threshold to filter far results."""
#     results = milvus_service.search_images(query)
    
#     # Initialize an empty list to hold parsed and filtered search results
#     parsed_results = []

#     # Iterate through each result and extract the relevant fields (id, distance)
#     for result in results:
#         for entity in result:
#             if entity.distance <= threshold:  # Only include matches below the threshold
#                 parsed_result = {
#                     "image_id": entity.id,
#                     "distance": entity.distance,
#                 }
#                 parsed_results.append(parsed_result)

#     # If no results meet the threshold, return a message
#     if not parsed_results:
#         return {"message": "No results found within the given distance threshold."}

#     # Return the parsed results
#     return {"results": parsed_results}

import os
# from fastapi import FileResponse
from fastapi.responses import FileResponse

# FastAPI route (can be placed in your route file)
@router.get("/search/")
async def search_images(query: str):
    """Search for images and return the image files."""
    image_paths = milvus_service.search_images(query)

    # Return the actual image files
    return [FileResponse(path) for path in image_paths if os.path.exists(path)]
    # return None
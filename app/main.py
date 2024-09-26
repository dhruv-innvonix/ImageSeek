from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.v1.image_upload import router as upload_router
from app.api.v1.image_search import router as search_router

app = FastAPI()

# Mount the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(upload_router, prefix="/api/v1/images", tags=["Image Upload"])
app.include_router(search_router, prefix="/api/v1/images", tags=["Image Search"])

@app.get("/")
async def root():
    return {"message": "Welcome to the ImageSeek API!"}

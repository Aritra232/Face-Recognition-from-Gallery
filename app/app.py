from fastapi import FastAPI, UploadFile, File
import shutil
import os

from app.service_1.add_image import process_and_add
from app.service_3.search import search_similar
from app.service_2.indexer import rebuild_index

app = FastAPI()

UPLOAD_DIR = "D:/new_1_image/gallery"
TEMP_DIR = "D:/new_1_image/temp"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    process_and_add(path)
    return {"message": "Image uploaded and indexed"}


@app.post("/search/")
async def search(
    file1: UploadFile = File(...),
    file2: UploadFile = File(None),
    file3: UploadFile = File(None),
):
    files = [f for f in [file1, file2, file3] if f is not None]

    query_paths = []
    for file in files:
        path = os.path.join(TEMP_DIR, file.filename)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        query_paths.append(path)

    results = search_similar(query_paths)
    return {"matched_images": results}


# ✅ FIX 2: Call this endpoint whenever images are moved / deleted from gallery
@app.post("/rebuild/")
async def rebuild():
    """
    Re-indexes the gallery folder from scratch.
    Call this after moving or deleting images so stale FAISS entries are removed.
    """
    stats = rebuild_index(gallery_path=UPLOAD_DIR)
    return {"message": "Index rebuilt", **stats}
from fastapi import FastAPI, UploadFile, File
import os
from io import BytesIO

from app.model.model import get_face_embeddings_from_bytes
from app.service_3.search import search_similar
from app.service_2.indexer import rebuild_index

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

UPLOAD_DIR = os.path.join(BASE_DIR, "gallery")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/search/")
async def search(
    file1: UploadFile = File(...),
    file2: UploadFile = File(None),
    file3: UploadFile = File(None),
):
    files = [f for f in [file1, file2, file3] if f is not None]

    query_embeddings = []

    for file in files:
        contents = await file.read()
        file_stream = BytesIO(contents)

        embeddings = get_face_embeddings_from_bytes(file_stream)
        query_embeddings.extend(embeddings)

    results = search_similar(query_embeddings, gallery_path=UPLOAD_DIR)

    return {"matched_images": results}


@app.post("/rebuild/")
async def rebuild():
    stats = rebuild_index(gallery_path=UPLOAD_DIR)
    return {"message": "Index rebuilt", **stats}
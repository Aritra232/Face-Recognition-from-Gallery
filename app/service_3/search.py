import face_recognition
import numpy as np
from app.model.model import get_face_embeddings
from app.service_2.indexer import load_index
import os

THRESHOLD = 0.5


def search_similar(query_paths):
    index, face_to_image = load_index()

    if index.ntotal == 0:
        print("⚠️ Index is empty. Run init_index.py first.")
        return []

    matched_images = set()

    gallery_embeddings = np.array(
        [index.reconstruct(i) for i in range(index.ntotal)],
        dtype="float32"
    )

    for path in query_paths:
        query_embeddings = get_face_embeddings(path)

        if not query_embeddings:
            print(f"⚠️ No faces detected in: {path}")
            continue

        for query_emb in query_embeddings:
            distances = face_recognition.face_distance(gallery_embeddings, query_emb)

            for idx, dist in enumerate(distances):
                if dist < THRESHOLD:
                    img_path = face_to_image[idx]

                    if os.path.exists(img_path):
                        matched_images.add(img_path)

    return list(matched_images)
import os
import numpy as np
import face_recognition
from app.model.model import get_face_embeddings

THRESHOLD = 0.50 
REFINE_THRESHOLD = 0.41


def get_gallery_data(gallery_path):
    gallery_data = []
    for img in os.listdir(gallery_path):
        img_path = os.path.join(gallery_path, img)
        if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            embeddings = get_face_embeddings(img_path)
            for emb in embeddings:
                gallery_data.append((emb, img_path))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return gallery_data


def search_similar(query_embeddings, gallery_path):
    gallery_data = get_gallery_data(gallery_path)
    if not gallery_data:
        return []

    gallery_embeddings = np.array([emb for emb, _ in gallery_data], dtype="float32")
    face_to_image = [path for _, path in gallery_data]

    matched_images = set()
    strong_match_embeddings = []

    # STEP 1
    for query_emb in query_embeddings:
        distances = face_recognition.face_distance(gallery_embeddings, query_emb)

        for idx, dist in enumerate(distances):
            if dist < THRESHOLD:
                img_path = face_to_image[idx]
                if os.path.exists(img_path):
                    matched_images.add(img_path)

            if dist < REFINE_THRESHOLD:
                strong_match_embeddings.append(gallery_embeddings[idx])

    # STEP 2
    if strong_match_embeddings:
        refined_query = np.mean(
            np.vstack(query_embeddings + strong_match_embeddings),
            axis=0
        )

        distances = face_recognition.face_distance(gallery_embeddings, refined_query)

        for idx, dist in enumerate(distances):
            if dist < THRESHOLD:
                img_path = face_to_image[idx]
                if os.path.exists(img_path):
                    matched_images.add(img_path)

    return list(matched_images)
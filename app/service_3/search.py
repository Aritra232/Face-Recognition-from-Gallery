import os
import numpy as np
import face_recognition
from app.service_2.indexer import load_index, build_index_from_gallery, is_index_outdated

THRESHOLD = 0.46
REFINE_THRESHOLD = 0.41


def search_similar(query_embeddings, gallery_path):
    index, face_to_image = load_index()

    # 🔥 AUTO UPDATE INDEX IF GALLERY CHANGED
    if index.ntotal == 0 or is_index_outdated(gallery_path):
        print("♻️ Gallery changed → rebuilding index...")
        index, face_to_image = build_index_from_gallery(gallery_path)

    gallery_embeddings = np.array(
        [index.reconstruct(i) for i in range(index.ntotal)],
        dtype="float32"
    )

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
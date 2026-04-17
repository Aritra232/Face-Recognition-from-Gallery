import faiss
import pickle
import os
import numpy as np

DIM = 128

INDEX_FILE = "face_index.faiss"
MAP_FILE = "face_map.pkl"


def load_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(MAP_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(MAP_FILE, "rb") as f:
            face_to_image = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(DIM)
        face_to_image = []

    return index, face_to_image


def save_index(index, face_to_image):
    faiss.write_index(index, INDEX_FILE)
    with open(MAP_FILE, "wb") as f:
        pickle.dump(face_to_image, f)


def add_faces_to_index(embeddings, image_path):
    index, face_to_image = load_index()

    for emb in embeddings:
        index.add(emb.reshape(1, -1).astype("float32"))
        face_to_image.append(image_path)

    save_index(index, face_to_image)


def rebuild_index(gallery_path="gallery"):
    from model.model import get_face_embeddings

    new_index = faiss.IndexFlatL2(DIM)
    new_face_to_image = []

    removed = 0
    added = 0

    _, old_face_to_image = load_index()
    unique_paths = list(set(old_face_to_image))

    for img_path in unique_paths:
        if not os.path.exists(img_path):
            removed += 1
            print(f"🗑️ Removed stale entry: {img_path}")
            continue

        embeddings = get_face_embeddings(img_path)

        for emb in embeddings:
            new_index.add(emb.reshape(1, -1).astype("float32"))
            new_face_to_image.append(img_path)
            added += 1

    save_index(new_index, new_face_to_image)

    print(f"✅ Rebuild complete: {added} face vectors kept, {removed} stale images removed.")

    return {"kept": added, "removed": removed}


def build_index_from_gallery(gallery_path):
    from app.model.model import get_face_embeddings

    index = faiss.IndexFlatL2(DIM)
    face_to_image = []

    for img in os.listdir(gallery_path):
        img_path = os.path.join(gallery_path, img)

        if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        embeddings = get_face_embeddings(img_path)

        for emb in embeddings:
            index.add(emb.reshape(1, -1).astype("float32"))
            face_to_image.append(img_path)

    save_index(index, face_to_image)

    print("✅ Auto index built from gallery")

    return index, face_to_image
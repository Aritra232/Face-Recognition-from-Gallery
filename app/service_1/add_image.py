from app.model.model import get_face_embeddings
from app.service_2.indexer import add_faces_to_index

def process_and_add(image_path):
    embeddings = get_face_embeddings(image_path)

    if len(embeddings) == 0:
        print(f"❌ No faces found in {image_path}")
        return

    add_faces_to_index(embeddings, image_path)

    print(f"✅ Indexed {len(embeddings)} faces from {image_path}")
import face_recognition
import numpy as np

def get_face_embeddings(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return []

    return [np.array(enc, dtype="float32") for enc in encodings]
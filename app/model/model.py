import face_recognition
import numpy as np

def get_face_embeddings(image_path):
    return get_face_embeddings_from_path(image_path)

def get_face_embeddings_from_path(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    return [np.array(enc, dtype="float32") for enc in encodings]


def get_face_embeddings_from_bytes(file_bytes):
    image = face_recognition.load_image_file(file_bytes)
    encodings = face_recognition.face_encodings(image)

    return [np.array(enc, dtype="float32") for enc in encodings]
import face_recognition
import numpy as np


def _ensure_uint8_rgb(image):
    if image is None:
        raise ValueError("Could not decode image")

    arr = np.asarray(image)

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 255.0)
        else:
            arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 4:
            arr = arr[:, :, :3]
        elif channels == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif channels != 3:
            raise ValueError("Unsupported image channel format")
    else:
        raise ValueError("Unsupported image dimensions")

    return np.ascontiguousarray(arr, dtype=np.uint8)


def get_face_embeddings(image_path):
    return get_face_embeddings_from_path(image_path)

def get_face_embeddings_from_path(image_path):
    image = face_recognition.load_image_file(image_path)
    image = _ensure_uint8_rgb(image)
    encodings = face_recognition.face_encodings(image)

    return [np.array(enc, dtype="float32") for enc in encodings]


def get_face_embeddings_from_bytes(file_bytes):
    if hasattr(file_bytes, "seek"):
        file_bytes.seek(0)

    image = face_recognition.load_image_file(file_bytes)
    image = _ensure_uint8_rgb(image)
    encodings = face_recognition.face_encodings(image)

    return [np.array(enc, dtype="float32") for enc in encodings]
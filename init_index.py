import os

from app.service_1.add_image import process_and_add

gallery_path = "gallery"

for img in os.listdir(gallery_path):
    path = os.path.join(gallery_path, img)

    if path.lower().endswith((".jpg", ".jpeg", ".png")):
        process_and_add(path)

print("✅ Gallery indexing complete")
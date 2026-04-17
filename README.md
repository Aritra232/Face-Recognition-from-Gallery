# 🚀 AI Face Search & Smart Gallery System (FastAPI + FAISS + Face Recognition)

An AI-powered system that detects faces from images, indexes them using FAISS, and allows users to search for similar faces using 1–3 query images. Built with FastAPI for high-performance APIs and optimized for fast similarity search.

---

## 📌 Features

- Face detection and embedding extraction using face_recognition  
- Fast similarity search using FAISS  
- Multi-image query support (upload 1–3 images)  
- Continuous image upload and automatic indexing  
- Rebuild index to remove stale or deleted images  
- FastAPI REST API backend  
- Lightweight and scalable architecture  
- Real-time face matching from gallery  

---

## ⚙️ Setup Instructions

### 1. Clone Repository

git clone https://github.com/your-username/face-search-ai.git  
cd face-search-ai  

---

### 2. Install Dependencies

pip install fastapi uvicorn face_recognition numpy faiss-cpu python-multipart  

---

### 3. Initialize Gallery Index

Make sure your images are inside the `gallery/` folder, then run:

python init_index.py  

This will:
- Detect faces in all images  
- Generate embeddings  
- Create FAISS index  

---

### 4. Run Backend (FastAPI)

uvicorn app.app:app --reload  

API will be available at:  
http://127.0.0.1:8000/docs  

---

## 📡 API Endpoints

### 🔹 Upload Image

POST /upload/  

- Upload an image  
- Detect faces  
- Add embeddings to FAISS index  

Response:

{
  "message": "Image uploaded and indexed"
}

---

### 🔹 Search Similar Faces

POST /search/  

- Upload 1–3 images  
- Returns matched images from gallery  

Response:

{
  "matched_images": [
    "gallery/img1.jpg",
    "gallery/img2.jpg"
  ]
}

---

### 🔹 Rebuild Index

POST /rebuild/  

- Rebuilds FAISS index from existing gallery  
- Removes deleted/missing images  

Response:

{
  "message": "Index rebuilt",
  "kept": 120,
  "removed": 5
}

---

## 🧠 How It Works

1. User uploads an image  
2. Face is detected using face_recognition  
3. Each face is converted into a 128-dimensional embedding  
4. Embeddings are stored in FAISS index  
5. Query images are processed the same way  
6. Distance between embeddings is computed  
7. Similar faces are retrieved from the gallery  

---

## 🗄️ Storage

- FAISS index file: face_index.faiss  
- Mapping file: face_map.pkl  

Stores:
- Face embeddings  
- Image path references  

---

## 🧠 Model Details

- Face Detection & Encoding: face_recognition (dlib-based)  
- Embedding Size: 128  
- Similarity Metric: Euclidean Distance  
- Threshold: 0.5 (tunable for accuracy)  

---

## 🎯 Design Decisions

- FastAPI for high-performance backend APIs  
- FAISS for fast vector similarity search  
- face_recognition for reliable face embeddings  
- Modular architecture (model → services → API)  
- Lightweight storage using pickle  

---

## ⚡ Important Notes

- Only images with detectable faces are indexed  
- Images without faces are skipped  
- Run `/rebuild/` after deleting or moving images  
- Threshold can be adjusted for stricter/looser matching  

---

## 🧪 Demo Workflow

1. Add images to `gallery/`  
2. Run `python init_index.py`  
3. Start server using FastAPI  
4. Upload query images via `/search/`  
5. View matched results  

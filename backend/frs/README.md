# Face Recognition System Backend (FRS)
Welcome to the Face Recognition System (FRS) Backend, a robust and efficient face detection and recognition service powered by cutting-edge technologies.

## 🚀 Features
- Face Detection: Detects and localizes faces within images.
- Face Recognition: Identifies individuals using face embeddings.
- Keypoint Extraction: Extracts facial landmarks for additional processing.
- Fast and Scalable: Designed to handle large volumes of data with low latency.
- Modular Design: Easily extensible for future enhancements.

## 🛠️ Technology Stack
- Framework: FastAPI (Python)
- Machine Learning: PyTorch
- RDBMS: PostgreSQL
- Vector Database: Qdrant
- Containerization: Docker

## 🧰 Prerequisites
Ensure you have the following installed on your system:
- Python 3.10+
- Docker & Docker Compose
- NVIDIA CUDA Toolkit (for GPU support)

## 🏗️ Installation
1. Clone the Repository
  
  ```
  git clone https://github.com/rabbicse/face-app.git  
  cd face-app/backend/frs  
  ```
2. Set Up Environment Variables. Create a `.env` file in the project root with the following:

  ```
  DATABASE_URL=postgresql://user:password@localhost:5432/frs_db  
  SECRET_KEY=your-secret-key  
  ALGORITHM=HS256  
  ACCESS_TOKEN_EXPIRE_MINUTES=30  
  ```

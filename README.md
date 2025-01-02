# Face Recognition Application

## Overview

**Face Recognition Application** is a cutting-edge facial recognition application built with modern technologies. It supports face detection, keypoint extraction and embedding generation. The app is optimized for high performance and accuracy, designed for use in identity verification, security and other facial recognition applications.

## Features

- **Face Detection**: Detect faces in real-time or static images.
- **Facial Landmarks Extraction**: Identify key facial points for advanced facial analysis.
- **Embedding Generation**: Generate facial embeddings for comparison and recognition.
- **API Support**: Provides RESTful APIs for integration.

## Technologies Used

- **Frontend**: React.js, Nextjs
- **Backend**: Python (FastAPI)
- **Face Detection - Backend**: PyTorch, RetinaFace
- **Face Recognition - Backend**: PyTorch, ArcFace
- **Face Detection - Frontend**: Tensorflowjs, Mediapipe
- **Database**: Qdrant (Vector Database), PostgreSQL (Optional for storage)
- **Containerization**: Docker

## Installation

### Prerequisites

Ensure you have the following installed:

- Node.js (v16 or higher)
- Python (v3.9 or higher)
- Docker (Optional, for containerized setup)
- CUDA Toolkit (Optional, for GPU acceleration)

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/rabbicse/face-app.git
    cd face-app
    ```

2. Install dependencies for the frontend:

    ```bash
    cd frontend
    npm install
    ```

3. Set up the backend:

    ```bash
    cd ../backend
    pip install -r requirements.txt
    ```

4. Configure environment variables:

    - Create a `.env` file in the root directory.
    - Add the following variables:

      ```env
      NEXT_PUBLIC_API_HOST=http://localhost
      NEXT_PUBLIC_API_PORT=5000
      ```

5. Start the services:

    - Frontend:

      ```bash
      cd frontend
      npm run dev
      ```

    - Backend:

      ```bash
      cd ../backend
      uvicorn api.server:app --reload --host 0.0.0.0 --port 5000
      ```

6. Access the application:

    Open your browser and navigate to `http://localhost:3000`.

## Screenshots

### Login Page

![Login Page](./screenshots/user-login.png)

### Login with Face Recognition

![Webcam Access](./screenshots/webcam-access.png)

## API Endpoints

### 1. Detect Faces

**Endpoint:** `/api/v1/dnn/detect`

- **Method:** `POST`
- **Body:** `multipart/form-data`
  - `photo`: The image file for detection.
- **Response:**

  ```json
  {
    "status": "success",
    "faces": [
      {
        "bbox": {
          "x_min": 0.2697347104549408,
          "y_min": 0.1822991967201233,
          "x_max": 0.7142632007598877,
          "y_max": 0.7912323474884033,
          "score": 0.9963470101356506
        },
        "landmarks": {
          "left_eye": {
            "x": 0.3953465223312378,
            "y": 0.413769394159317
          },
          "right_eye": {
            "x": 0.5912481546401978,
            "y": 0.4187081754207611
          },
          "nose": {
            "x": 0.49022188782691956,
            "y": 0.5452162623405457
          },
          "left_lip": {
            "x": 0.4161381721496582,
            "y": 0.6275689005851746
          },
          "right_lip": {
            "x": 0.5787203907966614,
            "y": 0.6303802132606506
          }
        }
      }
    ]
  }
  ```

### 2. Extract Embeddings

**Endpoint:** `/api/v1/dnn/extract-embedding`

- **Method:** `POST`
- **Body:** `multipart/form-data`
  - `photo`: The image file for embedding extraction.
- **Response:**

  ```json
    {
      "status": 0,
      "embedding": "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"
  }
  ```

## Docker Setup

To run the application in Docker, follow these steps:

1. Build and start the containers:

    ```bash
    docker-compose up --build
    ```

2. Access the application at `http://localhost:3000`.

## Contribution

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/your-feature-name`.
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References
- [InsightFace](https://github.com/deepinsight/insightface)
- [tfjs-models](https://github.com/tensorflow/tfjs-models)

---

Made with ❤️ by [Mehedi Hasan](https://github.com/rabbicse)

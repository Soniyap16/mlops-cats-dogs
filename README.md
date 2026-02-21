MLOps Pipeline for Cats vs Dogs Classification

This project implements a complete end-to-end MLOps pipeline for an image classification model (Cats vs Dogs), 

covering:

Model development and experiment tracking

Model packaging and containerization

CI/CD automation

Cloud deployment on Azure

Monitoring and logging

🚀 Project Overview

This project builds a machine learning system that:

Preprocesses image data

Trains a baseline classification model

Tracks experiments using MLflow

Serves predictions via FastAPI

Packages the model in Docker

Automates build and deployment using GitHub Actions

Deploys to Azure Container Apps

Monitors performance and logs predictions

🧱 Project Structure
mlops-cats-dogs/
│
├── src/
│   ├── data/
│   │   ├── preprocess.py
│   │   └── sample_data.py
│   └── models/
│       ├── train_baseline.py
│       └── train.py
│
├── app/
│   └── main.py
│
├── data/
│   ├── raw.dvc
│   ├── processed.dvc
│   └── sample/
│
├── .github/workflows/
│   └── ci.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-api.txt
├── baseline_model.pkl
├── evaluation.py
└── README.md


🧪 M1: Model Development & Experiment Tracking
✔ Version Control

Git used for source code

DVC used for dataset versioning

✔ Model

Baseline model: Logistic Regression on flattened images

Model saved as: baseline_model.pkl

✔ Experiment Tracking

MLflow used to track:

parameters

metrics (accuracy)

model artifacts

📦 M2: Model Packaging & Containerization
✔ Inference API

FastAPI endpoints:

GET /health

POST /predict

GET /metrics

✔ Environment

Dependencies managed via:

requirements.txt

requirements-api.txt

✔ Docker

Dockerfile created

Container built and tested locally

API accessible at http://localhost:8000/docs

🔁 M3: Continuous Integration (CI)

GitHub Actions pipeline automatically:

Runs unit tests (pytest)

Builds Docker image

Pushes image to Docker Hub

Docker image:

soniyapushparaj/cats-dogs-api:latest
☁️ M4: Continuous Deployment (CD)
Deployment Target

Azure Container Apps

CI/CD Flow

On each push to main:

Build image

Push to Docker Hub

Deploy to Azure

Run smoke test

Live API

👉 https://cats-dogs-api.orangepebble-10defb0f.eastus2.azurecontainerapps.io

📊 M5: Monitoring & Logging
Application-level logging

Request logs

Latency tracking

Prediction logging

Metrics endpoint
GET /metrics

Returns:

{
  "total_requests": 12
}
Azure Monitoring

Logs available in Azure Container Apps dashboard.

🧪 Post-Deployment Evaluation

Script: evaluation.py

Used to evaluate model on sample data and compute accuracy:

python evaluation.py
🧰 Tech Stack

Python

FastAPI

scikit-learn

OpenCV

MLflow

DVC

Docker

GitHub Actions

Azure Container Apps

▶️ How to Run Locally
1️⃣ Install dependencies
pip install -r requirements-api.txt
2️⃣ Run API
uvicorn app.main:app --reload
3️⃣ Open Swagger UI
http://127.0.0.1:8000/docs
🐳 Run with Docker
docker build -t cats-dogs-api .
docker run -p 8000:8000 cats-dogs-api
🎯 Key Features

✔ End-to-end MLOps pipeline
✔ Reproducible training
✔ Automated CI/CD
✔ Cloud deployment
✔ Monitoring and logging

👩‍💻 Author

Soniya P

📌 Conclusion

This project demonstrates a complete MLOps lifecycle, from data versioning and model training to deployment and monitoring in a production-like environment.

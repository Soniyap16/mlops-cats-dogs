FROM python:3.10-slim

WORKDIR /app

# install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# copy API requirements
COPY requirements-api.txt .

# install Python dependencies
RUN pip install --no-cache-dir -r requirements-api.txt

# copy app code and model
COPY app/ ./app/
COPY baseline_model.pkl .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
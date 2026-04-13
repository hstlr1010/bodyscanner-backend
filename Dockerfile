FROM python:3.11-slim

# System deps for OpenCV + PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install 4D-Humans from GitHub
RUN pip install --no-cache-dir \
    git+https://github.com/shubham-goel/4D-Humans.git

COPY . .

# Create storage dirs
RUN mkdir -p meshes checkpoints

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

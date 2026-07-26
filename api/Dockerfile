FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV/Pillow if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project context to include src modules and models
COPY src/ /app/src/
COPY api/ /app/api/
COPY models/ /app/models/

# The models and data directory will be mounted or populated at runtime, 
# but let's ensure the folders exist
RUN mkdir -p /app/models /app/data/train /app/data/test /app/data/raw

# Expose the port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ============================================================
# ROADWATCH AI - CES117V CAPSTONE PROJECT
# HUGGING FACE DOCKER CONFIGURATION
# ============================================================

# Use Python 3.11 for the application runtime.
FROM python:3.11-slim

# Prevent unnecessary Python bytecode generation.
ENV PYTHONDONTWRITEBYTECODE=1

# Enable immediate Python logging.
ENV PYTHONUNBUFFERED=1

# Set the application working directory.
WORKDIR /app

# Install system libraries required for video processing.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask application and model weights.
COPY . .

# Create writable application directories.
RUN mkdir -p uploads outputs

# Create a non-root application user.
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

USER user

# Hugging Face Spaces will use port 7860.
EXPOSE 7860

# Start the Flask app using Gunicorn.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "2", "--timeout", "300", "app:app"]

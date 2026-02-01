FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from the project root
# We use the same requirements file for simplicity in this monorepo-like structure
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the client directory
COPY client/ /app/client/
# Copy the entry point
COPY main.py /app/main.py

# Set PYTHONPATH to include the app directory
ENV PYTHONPATH=/app

# Verify the environment (optional, for debugging)
RUN python -c "import sys; print(sys.path)"

# The client entry point
CMD ["python", "main.py"]

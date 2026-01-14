FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from the project root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server directory into /app/server
COPY server/ /app/server/

# Set PYTHONPATH to include /app/server so that imports like 'from services...' work
ENV PYTHONPATH=/app/server

# Default command
CMD ["python", "server/main.py"]

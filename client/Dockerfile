FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from the project root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the client code
COPY client/ /app/client/

# Set PYTHONPATH so 'client.xxx' imports work
ENV PYTHONPATH=/app

# Entry point points to the new client-specific file
CMD ["python", "client/main.py"]

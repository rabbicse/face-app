# Use an official Python image as the base
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy the application files
COPY src /app/src

COPY models /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the application's port
EXPOSE 8000

# Set the working directory to the src directory
WORKDIR /app/src

# Command to run the application
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]

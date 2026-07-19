# ---------------------------------------------------------
# 1. BASE IMAGE
# ---------------------------------------------------------
# Use the official Python 3.10 slim image (lightweight Debian Linux)
FROM python:3.12-slim

# ---------------------------------------------------------
# 2. ENVIRONMENT VARIABLES
# ---------------------------------------------------------
# Prevents Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE 1
# Ensures Python output is sent straight to terminal (no buffering)
ENV PYTHONUNBUFFERED 1

# ---------------------------------------------------------
# 3. WORK DIRECTORY
# ---------------------------------------------------------
# Create and set the working directory inside the container
WORKDIR /app

# ---------------------------------------------------------
# 4. INSTALL DEPENDENCIES
# ---------------------------------------------------------
# Copy ONLY the requirements file first to leverage Docker layer caching.
# If you don't change requirements, Docker won't reinstall everything on the next build.
COPY requirements.txt .

# Install dependencies (no-cache-dir keeps the image small)
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------
# 5. COPY APPLICATION CODE & MODEL
# ---------------------------------------------------------
# Copy the FastAPI script and the serialized pipeline artifact
COPY main.py .
COPY sba_pipeline.joblib .

# ---------------------------------------------------------
# 6. EXPOSE PORT & RUN
# ---------------------------------------------------------
# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Command to run the application using Uvicorn
# We bind to 0.0.0.0 so it can accept traffic from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
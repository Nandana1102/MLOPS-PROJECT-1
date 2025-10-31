# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose port (Render uses 10000 by default)
EXPOSE 10000

# Set environment variables
ENV PORT=10000
ENV MODEL_PATH=models/model.joblib

# Run the Flask app
CMD ["python", "app/app.py"]

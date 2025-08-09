FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=predict_api/app.py
ENV PYTHONPATH=/app

# Run the application
CMD ["python", "predict_api/app.py"]
# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file first to utilize Docker build cache
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the application port
EXPOSE 5000

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Start the Flask app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "frontend.app:app"]

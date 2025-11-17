FROM python:3.9-slim

# Install required OS packages
RUN apt update -y && apt install -y awscli git

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Render
EXPOSE 10000

# Run the Flask app
CMD ["python", "app.py"]

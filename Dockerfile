# Dockerfile

# Runtime
FROM python:3.10-slim

# Working directory of the application
WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Make port 8080 available to the world
EXPOSE 8080

# Run app.py when the container launches
CMD ["python", "app.py"]
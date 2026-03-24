# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Create dummy directories to allow mount points for inputs and outputs if needed
RUN mkdir -p /app/input /app/output

# Run main.py when the container launches
CMD ["python", "main.py"]

# Use the official Triton Inference Server container as a base
# Using a 24.04 tag as an example, you can choose a different version
FROM nvcr.io/nvidia/tritonserver:24.04-py3

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install the Python packages first
# This leverages Docker's layer caching, so dependencies are only re-installed if requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Install FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
# Copy your entire model repository into the container's /models directory
# This is where Triton will look for models to serve.
COPY ./v2t /models

# Expose the ports for HTTP, gRPC, and metrics
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

# Set the command to start the Triton server and point it to the model repository
CMD ["tritonserver", "--model-repository=/models"]
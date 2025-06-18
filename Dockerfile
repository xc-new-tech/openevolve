# Use an official Python image as the base
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies including C/C++ toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    clang \
    clang++ \
    libc6-dev \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --root-user-action=ignore -e .

# Expose the project directory as a volume
VOLUME ["/app"]

# Set the entry point to the openevolve-run.py script
ENTRYPOINT ["python", "/app/openevolve-run.py"]
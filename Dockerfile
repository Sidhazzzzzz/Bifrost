FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install curl and install Ollama
RUN apt-get update && apt-get install -y curl zstd && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama daemon in background and bake the model into the image
RUN nohup bash -c "ollama serve &" && sleep 5 && ollama pull gemma2:2b

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and entrypoint
COPY app/ ./app/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create input/output/cache directories for the batch and demo contracts
RUN mkdir -p /input /output /cache

# Expose port for server mode
EXPOSE 8000

# Default: batch mode (reads /input/tasks.json → writes /output/results.json)
# Override with: --serve for interactive demo mode
ENTRYPOINT ["./entrypoint.sh"]

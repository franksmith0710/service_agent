FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/chroma /app/logs

# Make scripts executable
RUN chmod +x /app/scripts/docker-entrypoint.sh
RUN sed -i 's/\r$//' /app/scripts/docker-entrypoint.sh

# Expose ports
EXPOSE 8000 8501

# Default command will be overridden by docker-compose
CMD ["/app/scripts/docker-entrypoint.sh"]
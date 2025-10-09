FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create supervisor configuration
RUN mkdir -p /var/log/supervisor /etc/supervisor/conf.d
COPY supervisor.conf /etc/supervisor/conf.d/app.conf

# Expose port
EXPOSE 8000

# Run both API server and RQ worker using supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/app.conf"]

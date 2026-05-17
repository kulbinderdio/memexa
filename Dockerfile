FROM python:3.12-slim

WORKDIR /app

# System dependencies required by Playwright's Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl gnupg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright Chromium + its OS-level dependencies (libnss, libatk, etc.)
RUN playwright install chromium --with-deps

# Application source
COPY . .

EXPOSE 7700

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "7700"]

FROM python:3.10-slim

# Allow pip more time & retries on slow networks
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=5

# 1) install deps
COPY backend/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

# 2) copy code & models
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

WORKDIR /app/backend
EXPOSE 8000

CMD ["python", "app.py"]

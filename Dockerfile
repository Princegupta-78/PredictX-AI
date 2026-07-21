FROM python:3.11-slim

RUN apt-get update && apt-get install -y libgomp1 libstdc++6 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY requirements-frontend.txt .
RUN pip install --no-cache-dir --default-timeout=600 --retries=10 -r requirements.txt -r requirements-frontend.txt

COPY . .

ENV PYTHONPATH=/app:/app/ml
RUN chmod +x start.sh

EXPOSE 8501
CMD ["./start.sh"]
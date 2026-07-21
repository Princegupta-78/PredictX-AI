FROM python:3.10-slim

WORKDIR /app

# Copy both requirements files based on your structure
COPY requirements.txt .
COPY requirements-frontend.txt .

# Install everything together
RUN pip install --no-cache-dir -r requirements.txt -r requirements-frontend.txt

# Copy the rest of your project into the container
COPY . .

# Make the startup script executable
RUN chmod +x start.sh

# Run the startup script
CMD ["./start.sh"]
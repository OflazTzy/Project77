# Gunakan base image Python yang stabil (misalnya Python 3.10)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Salin semua file ke dalam container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Jalankan aplikasi dengan Gunicorn
CMD ["gunicorn", "app:app"]

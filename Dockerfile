# Imagine oficială Python cu suport pentru TensorFlow
FROM python:3.10-slim

# Setează directorul de lucru
WORKDIR /app

# Copiază fișierele în container
COPY . .

# ✅ Afișează ce fișiere sunt în director
RUN ls -la
# Instalează dependențele
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expune portul pentru Railway
EXPOSE 8080

# Rulează serverul cu gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]


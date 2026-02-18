# Imagen oficial de Airflow como base
FROM python:3.9-slim

WORKDIR /app 

# Ruta de las librerias necesarias para el proyecto
COPY requirements.txt .

# Instalamos las librerías necesarias 
RUN pip install --no-cache-dir -r /requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "scr.model_deploy:app", "--host=127.0.0.1", "--port=8000"]

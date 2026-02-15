# Imagen oficial de Airflow como base
FROM python:3.9-slim

WORKDIR /app 

# Ruta de las librerias necesarias para el proyecto
COPY Base_de_datos.xlsx ./data/
COPY /mlops_pipeline/scr/cargar_datos.py .
COPY model.pkl ./model/
COPY /mlops_pipeline/scr/model_deploy.py .
COPY requirements.txt .

# Instalamos las librerías necesarias 
RUN pip install --no-cache-dir -r /requirements.txt

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]

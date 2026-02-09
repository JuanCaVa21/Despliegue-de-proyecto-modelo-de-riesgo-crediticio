# Imagen oficial de Airflow como base
FROM apache/airflow:2.7.1

# Ruta de las librerias necesarias para el proyecto
COPY requirements.txt /requirements.txt

# Instalamos las librerías necesarias 
RUN pip install --no-cache-dir -r /requirements.txt
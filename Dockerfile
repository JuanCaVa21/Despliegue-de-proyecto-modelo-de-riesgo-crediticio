# Usamos la imagen oficial de Airflow como base
FROM apache/airflow:2.7.1

# Copiamos tu archivo de requerimientos al contenedor
COPY requirements.txt /requirements.txt

# Instalamos las librerías necesarias (incluyendo las de tu env_M5)
RUN pip install --no-cache-dir -r /requirements.txt
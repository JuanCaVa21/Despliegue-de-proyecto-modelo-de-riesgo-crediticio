from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pickle
import numpy as np
from cargar_datos import cargar_datos



app = FastAPI(
    title= 'Prediccion de reisgo crediticio',
    description= 'Predecir el riesgo crediticio mediante un modelo entrenado de XGBoost',
    version='1.0.0'
)

try:
    # Cargar modelo
    with open("../models/model.pkl", "rb") as f:
        model = pickle.load(f)
    print('Modelo Cargado')
except Exception as e:
    print('Error al cargar el modelo')

@app.get('/Saludo')
def Saludo():
    return {'Mensaje': 'Bienvenido a la API de prediccion'}

class Cliente_data(BaseModel):
    salario_cliente : int
    edad_cliente : int
    plazo_meses : int
    cuota_pactada : int
    tipo_credito : int
    deuda_total : float
    ingreso_disponible : int
    ratio_endeudamiento : float
    saldo_total : int
    cant_creditosvigentes: int
    creditos_sectorFinanciero : int
    creditos_sectorCooperativo : int
    creditos_sectorReal : int
    tipo_laboral : str
    tendencia_ingresos : str

@app.post('/Predict')
def Predict(cliente: Cliente_data):
    features = np.array(
        [
            [
                cliente.salario_cliente, 
                cliente.edad_cliente,
                cliente.plazo_meses,
                cliente.cuota_pactada,
                cliente.tipo_credito,
                cliente.deuda_total,
                cliente.ingreso_disponible,
                cliente.ratio_endeudamiento,
                cliente.saldo_total,
                cliente.cant_creditosvigentes,
                cliente.creditos_sectorFinanciero,
                cliente.creditos_sectorCooperativo,
                cliente.creditos_sectorReal,
                cliente.tipo_laboral,
                cliente.tendencia_ingresos
            ]
        ]
    )
    prediction = model.predict(features)[0] 
    return prediction

if __name__ == '__main__':
    uvicorn.run('model_deploy:app', port=8000, reload=True)
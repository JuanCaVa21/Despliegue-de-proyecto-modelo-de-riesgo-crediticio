import pandas as pd
import numpy as np
import optuna
import warnings

from cargar_datos import cargar_datos
from ft_engineering import preprocessor_data

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore') # Ignoramos las warnings

categoric_features = ['tipo_laboral']
ordinal_features = ['tendencia_ingresos']

try:
    # Cargamos el dataframe
    df = cargar_datos()
    print(f'Se cargaron correctamente: {df.shape}')

    df[categoric_features] = df[categoric_features].astype(object).replace('nan', 'NaN') # Correcion del tipo de dato
    df[ordinal_features] = df[ordinal_features].astype(str)

    # Hacemos una comprobacion de la variable objetivo
    if 'Pago_atiempo' in df.columns:
        X = df.drop(columns=['Pago_atiempo']).copy() # Variables features
        y = df['Pago_atiempo']                       # Variable target
        print('Se separaron las variables correctamente')
    else: 
        raise ValueError('La columna target no se encontro')
    
    df.dropna(subset=['Pago_atiempo']) # Limpieza variable target

    # Separacion dataset en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20,
        random_state=42,
        stratify=y)
    
except Exception as e:
    print(f'Error en Preprocesamiento: {e}')

def objective_rf(trial):
    """
    Funcion para obtener los hiperparametros deseados para el modelo de RandomForest

    Primero seleccionamos los parametros que queremos estudiar. Para este caso:

    n_estimators: El valor de arboles a entrenar (Entre 200 y 800)
    max_depth: Profundidad maxima de las caracteristicas consideradas por division
    min_samples_split: Minimo de muestras por Split
    min_samples_leaf: Minimo de muestras por hoja
    max_features: Cantidad maxima de caracteristicas consideradas por division

    Los demas parametros es recomdable dejarlos asi 
    class_weight: El peso que tiene cada caracteristica por division
    random_state: Para hacerlo reproducible
    n_jobs: Cuantos procesadores puede usar el Modelo

    Creamos las probabilidades para poder conocer el AUC Score y con eso definir cual es el mejor resultado de los Hiperparametros
    """

    rf_params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }

    model = RandomForestClassifier(**rf_params)

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor_data()),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    val_probs = pipe.predict_proba(X_train)[:, 1]
    auc_objetive_rf = roc_auc_score(y_train, val_probs)

    return auc_objetive_rf

if __name__ == "__main__":
    # Creamos un estudio con Optuna para empezar a intentar los hiperparametros
    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(objective_rf, n_trials=20) # 50 Intentos en el modelo 

    print(study_rf.best_params)

    # Modelo Final
    model_final = RandomForestClassifier(**study_rf.best_params)
    preprocessor = preprocessor_data()

    # Pipeline del modelo final
    Pipelines_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ('model', model_final)
        ]
    )
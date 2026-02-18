import pandas as pd
import numpy as np
import optuna
import warnings
import pickle

from cargar_datos import cargar_datos
from ft_engineering import preprocessor_data, split_df

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, classification_report,
                             precision_recall_curve)

warnings.filterwarnings('ignore') # Ignoramos las warnings
optuna.logging.set_verbosity(optuna.logging.ERROR) # Para que solo aparezcan errores criticos

def best_treshold_train(model, X_train, y_train):
    """
    Funcion para devolver el mejor treshold para calcular las metricas
    
    :param model: modelo de ML
    :param X_train: Para prediccion de probabilidad
    :param y_train: Para las metricas

    returns:
        best_y_train_pred: Mejor prediccion de y_train con el treshold
    """

    # Predecimos probabilidad en X_train
    y_train_proba = model.predict_proba(X_train)[:, 1]

     # Utilizando la metrica del threshold
    precision, recall, thresholds_train = precision_recall_curve(y_train, y_train_proba)

    # Buscamos cual es el mejor valor para poder usar en nuestro modelo
    f1_scores = []

    for t in thresholds_train:
        y_pred_treshold = (y_train_proba >= t).astype(int)
        f1_scores.append(f1_score(y_train, y_pred_treshold))

    f1_scores = np.array(f1_scores)

    best_idx = np.argmax(f1_scores)
    best_t = thresholds_train[best_idx]

    print(f"Best threshold: {best_t:.3f}")

    # Predicemos con el mejor Threshold que nos dio el anterior calculo
    best_y_train_pred = (y_train_proba >= best_t).astype(int)

    return best_y_train_pred, best_t

def best_treshold_test(model, X_test, y_test, best_t_train):
    """
    Funcion para devolver el mejor treshold para calcular las metricas
    
    :param model: modelo de ML
    :param X_test: Para prediccion de probabilidad
    :param y_test: Para las metricas

    returns:
        best_y_test_pred: Mejor prediccion de y_test con el treshold
    """

    # Predecimos probabilidad en X_test
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Predicemos con el mejor Threshold que nos dio el anterior calculo
    best_y_test_pred = (y_test_proba >= best_t_train).astype(int)

    return best_y_test_pred

def model_evaluation_metrics(model, X_train, X_test, y_train, y_test):
    """
    Genera metricas para vizualizar la viabilidad del modelo
    
    :param model: -> (...) model classification
    :param X_train: 
    :param X_test: 
    :param y_train: 
    :param y_test: 

    returns:
        metricas: Para 'train' y 'test'
    """

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    best_y_train_pred, best_t = best_treshold_train(model, X_train, y_train)
    best_y_test_pred = best_treshold_test(model, X_test, y_test, best_t)

    metricas = {
        'train': {
            'accuracy': accuracy_score(y_train, best_y_train_pred),
            'precision': precision_score(y_train, best_y_train_pred, zero_division=0),
            'recall': recall_score(y_train, best_y_train_pred, zero_division=0),
            'f1': f1_score(y_train, best_y_train_pred, zero_division=0),
            'auc_score': roc_auc_score(y_train, y_train_proba) 
        },
        'test': {
            'accuracy': accuracy_score(y_test, best_y_test_pred),
            'precision': precision_score(y_test, best_y_test_pred, zero_division=0),
            'recall': recall_score(y_test, best_y_test_pred, zero_division=0),
            'f1': f1_score(y_test, best_y_test_pred, zero_division=0),
            'auc_score': roc_auc_score(y_test, y_test_proba)
        }
    }

    overfitting_auc = metricas['test']['auc_score'] - metricas['train']['auc_score']

    # Diagnóstico de overfitting
    if overfitting_auc > 0.10:
        print(f"Overfitting detectado (diferencia AUC: {overfitting_auc:.4f})")
    elif overfitting_auc > 0.05:
        print(f"Overfitting moderado detectado (diferencia AUC: {overfitting_auc:.4f})")
    else:
        print(f"Modelo bien (diferencia AUC: {overfitting_auc:.4f})")

    print(metricas['test']['auc_score'])
    
    print(f"\nMatriz de Confusión (Test):")
    print(confusion_matrix(y_test, best_y_test_pred))
    
    print(f"\nReporte de Clasificación (Test):")
    print(classification_report(y_test, best_y_test_pred))
    
    return metricas

def objective_rf(trial, X_train, y_train):
    """
    Funcion para obtener los hiperparametros deseados para el modelo de XGBoost

    Primero seleccionamos los parametros que queremos estudiar. Para este caso:

    n_estimators: El valor de arboles a entrenar (Entre 100 y 600)
    max_depth: Profundidad maxima de cada arbol
    learning_rate: Cuanto contribuye cada nuevo modelo al resultado final
    subsample: Fraccion de observaciones a muestrear aleatoreamente
    colsample_bytree: Fraccion de caracteristicas a muestrear
    reg_alpha: L1 Promueve modelos mas simples 
    reg_lambda: L2 Controla el tamaño de los pesos

    Los demas parametros es recomdable dejarlos asi 
    objective: Sera la forma final que se determina el target
    random_state: Para hacerlo reproducible

    Creamos las probabilidades para poder conocer el AUC Score y con eso definir cual es el mejor resultado de los Hiperparametros
    """

    xgb_params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42
    }

    model = XGBClassifier(**xgb_params)

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor_data()),
        ("model", model)
    ])

    # Aplicamos validacion cruzada para mas efectividad
    cv_evaluation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_score = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=cv_evaluation,
        scoring='roc_auc'
    )

    mean_cv = cv_score.mean()

    return mean_cv

def entrenar_modelo(best_params, X_train, y_train):
    """
    Funcion para entrenar el modelo
    
    :param best_params: -> optuna.study.best_params pasa como kwargs al modelo
    :param X_train: 
    :param y_train: 

    return: 
        pipe_final: Pipeline entrenado con el modelo seleccionado
    """

    model_final = XGBClassifier(**best_params)

    pipe_final = Pipeline(
        steps=[
            ("preprocessor", preprocessor_data()),
            ("model_rf", model_final)
        ]
    )

    pipe_final.fit(X_train, y_train)

    return pipe_final

def save_model(model, threshold ,filename="model.pkl"):
    """
    Guarda el modelo final en un archivo .pkl 
    
    :param model: Modelo final
    :param filename: Nombre del archivo a guardar
    """
    bundle = {
        "pipeline": model,
        "threshold": threshold
    }

    with open(filename, 'wb')as f:
        pickle.dump(bundle, f)

if __name__ == "__main__":

    try:
        # Cargamos el dataset con cargar_datos()
        df = cargar_datos()

        # Separamos variables
        X_train, X_test, y_train, y_test = split_df(df, test_size=0.2, random_state=42)

        # Creamos un estudio con Optuna para empezar a intentar los hiperparametros
        study_rf = optuna.create_study(direction="maximize", study_name="rf_optuna", sampler=optuna.samplers.TPESampler(seed=42))

        # Optimizamos el estudio de optuna
        study_rf.optimize(lambda trial: objective_rf(trial, X_train, y_train), n_trials=20, show_progress_bar=True) # 20 intentos del modelo

        # Print a los parametros
        for params, value in study_rf.best_params.items():
            print(f'{params}: {value}') 
        print(f"AUC : {study_rf.best_value:.4f}")

        # Entrenamos el modelo con los hiperparametros que nos regreso optuna
        model_final = entrenar_modelo(study_rf.best_params, X_train, y_train)

        # Mostramos las metricas de evaluacion
        model_evaluation_metrics(model_final, X_train, X_test, y_train, y_test)

        # Obtenemos el threshold óptimo sobre train para guardarlo
        _, best_t = best_treshold_train(model_final, X_train, y_train)
        print(f"\nThreshold óptimo guardado: {best_t:.4f}")

        # Guardamos el modelo .pkl junto con su threshold
        save_model(model_final, best_t, "model.pkl")
        
    except Exception as e:
        print(f'Error {e}')
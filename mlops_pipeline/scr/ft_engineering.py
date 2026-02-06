# Feature Enginnering para los datos 

import pandas as pd
import numpy as np
from mlops_pipeline.scr.cargar_datos import cargar_datos

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline as pipe

from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropFeatures


numeric_features = ['salario_cliente', 'capital_prestado', 'total_otros_prestamos', 'edad_cliente', 'plazo_meses', 'cuota_pactada', 'tipo_credito']
categoric_features = ['tipo_laboral']
ordinal_features = ['tendencia_ingresos']
drop_features = ['fecha_prestamo', 'puntaje', 'saldo_mora', 'saldo_mora_codeudor', 'huella_consulta']

def preprocessor_data():
    """
    Crea el pipeline ded preprocesamiento de datos.
    
    Returns:
        Pipeline: Pipeline de sklearn con todos los pasos del preprocesamiento
    """

    preprocessor_sk = ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), numeric_features),
            ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore'), categoric_features),
            ('ordinal', OrdinalEncoder(), ordinal_features)
        ]
    )

    Pipeline = pipe(
        steps= [
            ('drop_features', DropFeatures(features_to_drop = drop_features)),
            ('imputer_numeric', MeanMedianImputer(imputation_method='median', variables=numeric_features)),
            ('imputer_categorical', CategoricalImputer(imputation_method='frequent', variables=categoric_features + ordinal_features)),
            ('outliers', Winsorizer(capping_method='quantiles', tail='right', fold=0.05, variables=numeric_features)),
            ('preprocessor', preprocessor_sk)
        ]
    )

    return Pipeline

if __name__ == "__main__":
    try:
        # Cargamos el dataframe
        df = cargar_datos()
        print(f'Se cargaron correctamente: {df.shape}')

        df[categoric_features] = df[categoric_features].astype(object).replace('nan', 'NaN')
        df[ordinal_features] = df[ordinal_features].astype(str)

        if 'Pago_atiempo' in df.columns:
            X = df.drop(columns=['Pago_atiempo']).copy()
            y = df['Pago_atiempo']
            print('Se separaron las variables correctamente')
        else: 
            raise ValueError('La columna target no se encontro')
        
        df.dropna(subset=['Pago_atiempo'])

        Pipeline = preprocessor_data()
        X_transformed = Pipeline.fit_transform(X)

        total_nulos = np.isnan(X_transformed).sum()
        
        if total_nulos == 0:
            print(f'Cantidad total de valores nulos despues de la transformacion: {total_nulos}')
            print('Preprocesamiento OK')

        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, 
            test_size=0.20,
            random_state=42,
            stratify=y)
        
    except Exception as e:
        print(f'Error en Preprocesamiento: {e}')


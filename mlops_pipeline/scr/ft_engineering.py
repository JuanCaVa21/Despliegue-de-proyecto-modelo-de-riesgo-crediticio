# Feature Enginnering para los datos 

import pandas as pd
import numpy as np
from cargar_datos import cargar_datos
import warnings

warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline as pipe

from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropFeatures

def crear_features_adicionales(df):
    """
    Crea features que capturan relaciones importantes

    params:
        df: Dataframe original
    
    returns:
        df_copy: copia del dataframe de entrada pero con los features adicionales
    """
    
    df_copy = df.copy()
    # Ratio de endeudamiento (muy importante para riesgo crediticio)
    df_copy['ratio_endeudamiento'] = df_copy['cuota_pactada'] / (df_copy['salario_cliente'] + 1)
    
    # Ingreso disponible
    df_copy['ingreso_disponible'] = df_copy['salario_cliente'] - df_copy['cuota_pactada']
    
    # Deuda total
    df_copy['deuda_total'] = df_copy['capital_prestado'] + df_copy['total_otros_prestamos']

    df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
    
    return df_copy

def preprocessor_data():
    """
    Crea el pipeline ded preprocesamiento de datos.
    
    Returns:
        Pipeline: Pipeline de sklearn con todos los pasos del preprocesamiento
    """

    numeric_features = ['salario_cliente', 'edad_cliente', 'plazo_meses', 'cuota_pactada', 'deuda_total', 'ingreso_disponible', 'ratio_endeudamiento', 'saldo_total', 'cant_creditosvigentes', 'creditos_sectorFinanciero', 'creditos_sectorCooperativo', 'creditos_sectorReal']
    categoric_features = ['tipo_laboral']
    ordinal_features = ['tendencia_ingresos']
    drop_features = ['fecha_prestamo', 'puntaje', 'saldo_mora', 'saldo_mora_codeudor', 'huella_consulta', 'capital_prestado', 'total_otros_prestamos', 'puntaje_datacredito', 'saldo_principal', 'promedio_ingresos_datacredito', 'tipo_credito']

    preprocessor_sk = ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), numeric_features),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categoric_features),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features)
        ],
        remainder='passthrough'
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

def split_df(df, test_size=0.2, random_state=42):
    """
    train-test split del df

    Divide el dataset dependiendo de los valores que se asignen 

    Params:
        df: Dataframe a dividir
        test_size: Tamaño a dividir el test. default 80/20
        random_state: Hace que sea aleatoria la asignacion. default 42

    returns:
        X_train, X_test: Features en la proporcion indicada al llamar la funcion 
        y_train, y_test: Target en la proporcion indicada
    """

    # Se hace este paso para poder asi separar junto a los features nuevos
    df_split = crear_features_adicionales(df)

    # Cambiamos tipo de variables a categoricas
    df_split['tipo_laboral'] = df_split['tipo_laboral'].astype(str)
    df_split['tendencia_ingresos'] = df_split['tendencia_ingresos'].astype(str)

    # Hacemos limpieza de nulos en la variable target
    df_split = df_split.dropna(subset=['Pago_atiempo'])

    # Dividimos el dataser en target y features
    if 'Pago_atiempo' in df_split.columns: # Para verificar que target este en el df
        X = df_split.drop(columns=['Pago_atiempo']).copy() # Features
        y = df_split['Pago_atiempo'] # Target
        print(f'Se separaron las variables correctamente para X: {len(X)} para y: {len(y)}')
    else: 
        raise ValueError('La columna target no se encontro')
    
    # Aqui se crea el split en las proporciones
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state,
        stratify=y)
    
    print(f'El split de train quedo: {len(X_train)}')
    print(f'El split de test quedo {len(X_test)}')
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    try:
        # Cargamos el dataframe
        df = cargar_datos()
        print(f'Se cargaron correctamente: {df.shape}')

        
        X_train, X_test, y_train, y_test = split_df(df)
        
        Pipeline = preprocessor_data()
        Pipeline.fit_transform(X_train, y_train)

    except Exception as e:
        print(f'Error en Preprocesamiento: {e}')

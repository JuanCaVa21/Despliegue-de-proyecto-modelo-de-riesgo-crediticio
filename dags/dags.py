from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline as Pipe

from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropFeatures

### Posible error en este Script ####
def cargar_datos(**context):
    """
    Carga de los datos
    
    cargar_datos(..):
        
        Carga un excel en una ruta especifica
    """

    ruta_excel = '/Users/juanv/Documents/GitHub/Despliegue-de-proyecto-modelo-de-riesgo-crediticio/data/Base_de_datos.xlsx'

    # En caso de que el archivo no exista
    if not ruta_excel:
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta_excel}")

    # Creamos nuestra variable a invocar despues
    df = pd.read_excel(ruta_excel) 

    context['ti'].xcom_push(key='dataframe', value=df.to_json())
    
    return f"Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas"


def crear_preprocessor():
    """
    Crea el pipeline de preprocesamiento de datos.
    
    Returns:
        Pipeline: Pipeline de sklearn con todos los pasos de preprocesamiento
    """
    # Definición de features
    numeric_features = ['salario_cliente', 'capital_prestado', 'total_otros_prestamos', 'edad_cliente', 'plazo_meses', 'cuota_pactada', 'tipo_credito']
    categoric_features = ['tipo_laboral']
    ordinal_features = ['tendencia_ingresos']
    drop_features = ['fecha_prestamo', 'puntaje', 'saldo_mora', 'saldo_mora_codeudor', 'huella_consulta']

    # Transformador de datos
    preprocessor_sk = ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), numeric_features),
            ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore'), categoric_features),
            ('ordinal', OrdinalEncoder(), ordinal_features)
        ]
    )

    # Pipeline de datos
    Pipeline = Pipe(
        steps= [
            ('drop_features', DropFeatures(features_to_drop = drop_features)),
            ('imputer_numeric', MeanMedianImputer(imputation_method='median', variables=numeric_features)),
            ('imputer_categorical', CategoricalImputer(imputation_method='frequent', variables=categoric_features + ordinal_features)),
            ('outliers', Winsorizer(capping_method='quantiles', tail='right', fold=0.05, variables=numeric_features)),
            ('preprocessor', preprocessor_sk)
        ]
    )

    return Pipeline


def preprocessor_data(**context):
    """
    Ejecuta el preprocesamiento de datos.
    
    Recupera los datos del XCom y aplica el pipeline de preprocesamiento.
    """

    # Recuperar datos de la tarea anterior
    ti = context['ti']
    df_json = ti.xcom_pull(key='dataframe', task_ids='cargar_datos') # Idea de Gemini. Creo que tenemos que buscar una forma mas optima
    
    # Para ver si se pudieron recuperar datos
    if df_json is None:
        raise ValueError("No se pudieron recuperar los datos de la tarea anterior")
    
    # Convertir de JSON a DataFrame
    df = pd.read_json(df_json)
    
    # Crear pipeline de preprocesamiento
    pipeline = crear_preprocessor()
    
    # Separar features y target 
    X = df.drop('Pago_atiempo', axis=1)  
    y = df['Pago_atiempo']
    
    X_transformed = pipeline.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, 
    test_size=0.20,
    random_state=42,
    stratify=y)

    return f"Pipeline creado con {len(pipeline.steps)} pasos"


# Configuracion del DAG de Airflow
default_args = {
    'owner': 'JuanV', # Usuario
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 6),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='dag_riesgo_crediticio',  # Nombre del DAG
    default_args=default_args,
    description='Pipeline de carga y procesamiento de riesgo crediticio',
    schedule_interval=None,  
    catchup=False,
    tags=['riesgo', 'financiero', 'ml'], # Grupos para filtrar
    max_active_runs=1,  
) as dag:
    
    # Tarea 1: Cargar datos
    task_cargar_datos = PythonOperator(
        task_id='cargar_datos',
        python_callable=cargar_datos,
        provide_context=True, 
    )

    # Tarea 2: Feature Engineering
    task_ft_engineering = PythonOperator(
        task_id='ft_engineering',
        python_callable=preprocessor_data, 
        provide_context=True,  
    )
    
    task_cargar_datos >> task_ft_engineering
import streamlit as st
import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from ft_engineering import split_df, preprocessor_data
from scipy.stats import ks_2samp, chi2_contingency

model_path = '/Users/juanv/Documents/GitHub/Despliegue-de-proyecto-modelo-de-riesgo-crediticio/models/model.pkl'
data_path = '/Users/juanv/Documents/GitHub/Despliegue-de-proyecto-modelo-de-riesgo-crediticio/Base_de_datos.xlsx'

# Traemos el modelo entrenado
with open(model_path, 'rb')as f:
    loaded = pickle.load(f) # Guardamos el modelo en una 
    
# Si el pickle guardo un dict, extraemos el pipeline del modelo
if isinstance(loaded, dict):
    # Buscamos la llave que contenga el pipeline (la primera que no sea escalar)
    model = next(v for v in loaded.values() if hasattr(v, 'predict_proba'))
else:
    model = loaded

df = pd.read_excel(data_path)

X_train, X_test, y_train, y_test = split_df(df)
X_train.dropna()

procesado = preprocessor_data()
df_procesado = procesado.fit_transform(X_train, y_train)

# Creamos la funcion con la que vamos a predecir
def model_prediction(input_data: dict, model):
    """
    Construye un DataFrame con todas las columnas que el pipeline espera,
    incluyendo las que serán dropeadas internamente (se envían como np.nan).

    params:
        input_data: dict con los datos de entrada para la prediccion
        model: pipeline de sklearn con el modelo entrenado y guardado en model.pkl
        threshold: umbral para clasificar la prediccion (default=0.5)
    
    returns:
        y_pred: prediccion binaria (0 o 1) del modelo
        proba: probabilidad de la clase positiva (cliente pago)
    """
    # Columnas que el pipeline dropea: se pasan como NaN, el step DropFeatures las elimina
    drop_cols = {
        'fecha_prestamo': np.nan,
        'puntaje': np.nan,
        'saldo_mora': np.nan,
        'saldo_mora_codeudor': np.nan,
        'huella_consulta': np.nan,
        'capital_prestado': np.nan,
        'total_otros_prestamos': np.nan,
        'puntaje_datacredito': np.nan,
        'saldo_principal': np.nan,
        'promedio_ingresos_datacredito': np.nan,
        'tipo_credito': np.nan,
    }
    # Unimos los datos ingresados con las columnas dropeadas
    full_row = {**input_data, **drop_cols}
    df_input = pd.DataFrame([full_row])
    y_proba = model.predict_proba(df_input)[0, 1] # Probabilidad de la clase positiva (cliente pago o no)  
    return y_proba

# Funcion para clasificar y mostrar el resultado de la prediccion con un mensaje y la probabilidad
def classify(y_proba):
    if y_proba == 1: # Pago a tiempo = 1
        st.markdown("<h3 style='color: green;'>Cliente Pago 😊</h3>", unsafe_allow_html=True)
    else: # Pago a tiempo = 0
        st.markdown("<h3 style='color: green;'>Cliente en riesgo 😒</h3>", unsafe_allow_html=True)
    st.caption(f"Probabilidad de pago: **{y_proba:.1%}**")

st.set_page_config(
    page_title= 'App interactiva del modelo de prediccion de riesgo crediticio',
    page_icon= '🤑',
    layout= 'centered',
    initial_sidebar_state= 'auto'
)

st.title('App de Monitoreo del modelo')

# tabs en streamlit para organizar la informacion en secciones y que el usuario pueda navegar entre ellas
tab_predecir, tab_load_data, tab_evaluar, tab_metricas = st.tabs(['Predecir', 'Cargar Datos', 'Evaluar Data Drift', 'Visualizacion'])

with st.sidebar:
    st.title('Menu de Opciones')
    st.markdown('Selecciona una opcion para empezar')
    tab_side_load_data, tab_side_metricas = st.tabs(['Cargar Datos', 'Visualizacion'])

# En este tab se muestra el formulario para ingresar los datos manualmente y hacer una prediccion puntual, se llama a la funcion model_prediction y classify para mostrar el resultado de la prediccion con un mensaje y la probabilidad.
with tab_predecir:
    st.write('Aqui puedes seleccionar los datos manualmente para predecir un caso puntual')

    salario_cliente            = st.number_input('Salario Cliente', min_value=1000000, max_value=10000000, step=100000, value=1000000, placeholder='Type a Number')
    edad_cliente               = st.slider('Edad Cliente', 18, 80, step=1, format="%d Años")
    plazo_meses                = st.slider('Plazo Meses', 1, 90, step=1, format="%d Meses")
    cuota_pactada              = st.number_input('Cuota Pactada', min_value=0, max_value=4000000, step=1, value=None, placeholder='Type a Number')
    deuda_total                = st.number_input('Deuda Total (Capital Prestado + Otros Prestamos)', min_value=300000, max_value=50000000, step=1, value=300000, placeholder='Type a Number')
    ingreso_disponible         = st.number_input('Ingreso Disponible (Ingresos - Cuota Pactada)', min_value=500000, max_value=10000000, step=1, value=500000, placeholder='Type a Number')
    ratio_endeudamiento        = st.number_input('Ratio Endedudamiento', placeholder='Type a Number', format="%0.001f", step=0.001)
    saldo_total                = st.number_input('Saldo Total', placeholder='Type a Number')
    cant_creditosvigentes      = st.slider('Cantidad de Creditos Vigentes', 1, 70, step=1)
    creditos_sectorFinanciero  = st.slider('Creditos Sector Financiero', 1, 70, step=1)  
    creditos_sectorCooperativo = st.slider('Creditos Sector Cooperativo', 1, 70, step=1)
    creditos_sectorReal        = st.slider('Creditos Sector Real', 1, 70, step=1)
    tipo_laboral               = st.selectbox('Tipo de Trabajador', ('Empleado', 'Independiente'))
    tendencia_ingresos         = st.selectbox('Tendencia de los ingresos del Cliente', ('Creciente', 'Decreciente', 'Estable'))

    if st.button('Hacer Prediccion'):
        # Se construye un dict con los nombres exactos de columna que el pipeline conoce 
        input_data = {
            'salario_cliente':            int(salario_cliente),
            'edad_cliente':               int(edad_cliente),
            'plazo_meses':                int(plazo_meses),
            'cuota_pactada':              int(cuota_pactada) if cuota_pactada is not None else 0,
            'deuda_total':                int(deuda_total),
            'ingreso_disponible':         int(ingreso_disponible),
            'ratio_endeudamiento':        float(ratio_endeudamiento),
            'saldo_total':                float(saldo_total),
            'cant_creditosvigentes':      int(cant_creditosvigentes),
            'creditos_sectorFinanciero':  int(creditos_sectorFinanciero),
            'creditos_sectorCooperativo': int(creditos_sectorCooperativo),
            'creditos_sectorReal':        int(creditos_sectorReal),
            'tipo_laboral':               str(tipo_laboral),
            'tendencia_ingresos':         str(tendencia_ingresos),
        }

        y_pred = model_prediction(input_data, model)
        classify(y_pred)

# tab para cargar un dataset nuevo, se muestra el dataframe cargado y se evalua el data drift con respecto al dataset original con el que se entreno el modelo.
with tab_side_load_data:
    st.subheader('Cargar Dataset')
    st.write('Para cargar el dataset recuerda que debe ser en formato CSV (delimited by comma) o Excel (.xlsx)')

    dataset_nuevo = st.file_uploader(label='Upload dataset', type=["csv", "xlsx"])

    if dataset_nuevo is not None:
        if dataset_nuevo.name.endswith('.csv'):
            df_nuevo = pd.read_csv(dataset_nuevo)
            st.success('CSV Cargado Correctamente')
        elif dataset_nuevo.name.endswith('.xlsx'):
            df_nuevo = pd.read_excel(dataset_nuevo)
            st.success('Excel Cargado Correctamente')
        else:
            st.error('Formato no soportado')

    X_train_n, X_test_n, y_train_n, y_test_n = split_df(df_nuevo)

    procesado = preprocessor_data()
    df_nuevo_procesado = procesado.fit_transform(X_train_n, y_train_n)

with tab_load_data:
    if dataset_nuevo is not None:
        st.write('El dataframe cargado:')
        st.dataframe(df_nuevo.head(5))
    else:
        st.write('Carga un dataset para ver sus datos')

# tab del dashboard de visualizacion de metricas, se pueden mostrar graficas como la curva ROC, la matriz de confusion, la distribucion de las probabilidades predichas, etc.
with tab_side_metricas:
    st.write('Desde este menu puedes controlar la visualizacion del dashboard de los resultados de las predicciones y de los datos')
    
    st.subheader('Métricas de Evaluación')
    select = st.selectbox('Selecciona una métrica', ['Precision - Recall', 'ROC Curve', 'Matriz de Confusion'])
    
with tab_metricas:
    st.write('Gráficas y Visualización de métricas obtenidas') 
    
    # Determinar qué datos usar para las métricas
    if X_test is not None and dataset_nuevo is not None:
        st.write(f"Mostrando métricas del set original en ({len(X_test)} muestras)")
        data_available = True
        # Verificamos que el target este en el df
        if 'Pago_atiempo' not in df_nuevo.columns:
            st.error("El dataset cargado no contiene la columna target")
            data_available = False
    else:
        st.warning("Carga un dataset en la pestaña 'Cargar Datos' para visualizar métricas.")
        data_available = False

    if data_available:
        try:
            # Probabilidades de y
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            # Porfin las visualizaciones
            if select == 'Precision - Recall':
                precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
                promedio_PR = average_precision_score(y_test, y_proba)
                
                fig, ax = plt.subplots()
                ax.plot(recall, precision, label=f'Promedio: {promedio_PR:.3f}')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve')
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            elif select == 'ROC Curve':
                False_Positive_Rate, True_Positive_Rate, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(False_Positive_Rate, True_Positive_Rate)

                fig, ax = plt.subplots()
                ax.plot(False_Positive_Rate, True_Positive_Rate, label=f'AUC = {roc_auc:.3f}')
                ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            elif select == 'Matriz de Confusion':

                matriz_de_confucion = confusion_matrix(y_test, y_pred)
                st.write('Matriz de Confusion: ', matriz_de_confucion)
                
        except Exception as e:
            st.error(f"Error al calcular métricas: {str(e)}")
            st.info("Asegúrate de que el dataset tenga las mismas columnas que el dataset de entrenamiento.")

# tab para evaluar el data drift del dataset cargado con respecto al dataset original, se pueden usar metricas como la distancia de Kolmogorov-Smirnov para variables numericas y la distancia de Jensen-Shannon para variables categoricas.
with tab_evaluar:
    if dataset_nuevo is not None:
        st.write('Evaluando Data Drift')

        X_train_clean = X_train.dropna()
        X_train_n_clean = X_train_n.dropna()

        # Separamos columnas numericas y categoricas
        cols_numericas   = X_train_clean.select_dtypes(include=[np.number]).columns.tolist()
        cols_categoricas = X_train_clean.select_dtypes(include=['object', 'category']).columns.tolist()

        cols_numericas = [c for c in cols_numericas if c in X_train_n_clean.columns]
        cols_categoricas = [c for c in cols_categoricas if c in X_train_n_clean.columns]

        st.subheader('KS Test (Kolmogorov-Smirnov) Para numericas')
        if cols_numericas:
            resultados_ks = []
            for col in cols_numericas:
                serie_original = X_train_clean[col].dropna()
                serie_nueva = X_train_n_clean[col].dropna()
                ks_stats, ks_pvalue = ks_2samp(serie_original, serie_nueva)
                resultados_ks.append({
                    'Variable' : col,
                    'KS Stat': ks_stats.round(3),
                    'P Value': ks_pvalue.round(3),
                    'Drift': 'Si' if ks_pvalue < 0.05 else 'No'
                })
            df_ks = pd.DataFrame(resultados_ks)
            st.dataframe(df_ks, use_container_width=True)

        st.subheader('Chi-Cuadrado para Categoricas')
        if cols_categoricas:
            resultados_chi = []
            for col in cols_categoricas:
                serie_original = X_train_clean[col].dropna()
                serie_nueva = X_train_n_clean[col].dropna()

                # Tabla de contingencia
                categorias = pd.Series(list(set(serie_original) | set(serie_nueva)))
                frecuencia_original = serie_original.value_counts().reindex(categorias, fill_value = 0)
                frecuencua_nueva = serie_nueva.value_counts().reindex(categorias, fill_value=0)
                contingencia = pd.DataFrame([frecuencia_original.values, frecuencua_nueva.values])

                chi_stats, chi_pvalue, _, _ = chi2_contingency(contingencia)
                resultados_chi.append({
                    'Variable' : col, 
                    'Chi2 Stats' : chi_stats.round(3),
                    'Chi2 Pvalue' : chi_pvalue.round(3),
                    'Drift': 'Si' if chi_pvalue < 0.05 else 'No'
                })
            df_chi = pd.DataFrame(resultados_chi)
            st.dataframe(df_chi, use_container_width=True)

    else:
        st.warning('Para evaluar drift es necesario 2 datasets')






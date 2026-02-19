# Modelo de Predicción de Riesgo Crediticio — Despliegue e Monitoreo

> Sistema de machine learning para predecir el comportamiento de pago de clientes crediticios, con una aplicación interactiva de monitoreo construida en Streamlit y un pipeline MLOps orquestado con Apache Airflow.

---

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Arquitectura](#arquitectura)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Tecnologías](#tecnologías)
- [Instalación](#instalación)
- [Uso](#uso)
- [App de Monitoreo](#app-de-monitoreo)
- [Pipeline MLOps](#pipeline-mlops)
- [Data Drift](#data-drift)
- [Licencia](#licencia)

---

## Descripción del Proyecto

Este proyecto desarrolla un **modelo predictivo de riesgo crediticio** mediante técnicas de aprendizaje automático, entrenado con información histórica de créditos para una empresa en el sector financiero. La empresa opera bajo un esquema estructurado de proyectos, en el cual cada iniciativa debe seguir una arquitectura de carpetas estrictamente definida. Esta estructura no puede ser modificada, ya que los procesos de despliegue a producción están automatizados a través de pipelines de validación en Jenkins. Cualquier alteración en la organización de carpetas podría generar retrasos significativos en el paso a producción.


El sistema incluye:
- Un **modelo de clasificación** entrenado y serializado con `pickle`.
- Una **aplicación interactiva** en Streamlit para predicciones puntuales y monitoreo del modelo.
- Módulos de **feature engineering** reutilizables integrados en un pipeline de `sklearn`.
- Una API de prediccion para poder ser consumida por otros sistemas.

---

## Arquitectura

```
Base_de_datos.xlsx
       │
       ▼
ft_engineering.py          ← Preprocesamiento y split de datos
       │
       ├──► model.pkl      ← Pipeline sklearn serializado (modelo entrenado)
       │
       └──► model_monitoring.py   ← App Streamlit
                │
                ├── Tab Predecir      → Predicción puntual por formulario
                ├── Tab Cargar Datos  → Upload de nuevo dataset
                ├── Tab Visualización → ROC, Precision-Recall, Matriz de Confusión
                └── Tab Data Drift    → KS Test + Chi-Cuadrado

dags/                      ← DAGs de Apache Airflow (orquestación MLOps)
mlops_pipeline/scr/        ← Scripts del pipeline de reentrenamiento
```

---

## Estructura del Repositorio

```
├── 📁 .devcontainer
│   └── ⚙️ devcontainer.json
├── 📁 dags
│   └── 🐍 dags.py
├── 📁 mlops_pipeline
│   └── 📁 scr
│       ├── 🐍 cargar_datos.py
│       ├── 📄 comprension_EDA.ipynb
│       ├── 🐍 ft_engineering.py
│       ├── 🐍 model_deploy.py
│       ├── 🐍 model_monitoring.py
│       └── 🐍 model_training_evaluation.py
├── 📁 models
│   └── 📄 model.pkl
├── ⚙️ .gitattributes
├── ⚙️ .gitignore
├── 📄 Base_de_datos.xlsx
├── 🐳 Dockerfile
├── 📄 LICENSE
├── 📝 README.md
├── 📄 model.pkl
└── 📄 requirements.txt
```

---

## Tecnologías

| Categoría | Herramientas |
|---|---|
| Lenguaje | Python 3.9 | Anaconda Enviorment |
| ML & Preprocesamiento | scikit-learn, pandas, numpy, feature-engine, optuna |
| Aplicación Web | Streamlit |
| Orquestación MLOps | Apache Airflow |
| Estadística | scipy (KS Test, Chi-Cuadrado) |
| Visualización | matplotlib |
| Contenerización | Docker |
| Serialización | pickle |

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/JuanCaVa21/Despliegue-de-proyecto-modelo-de-riesgo-crediticio.git
cd Despliegue-de-proyecto-modelo-de-riesgo-crediticio
```

### 2. Crear entorno virtual en Anaconda

```bash
conda create -n env_riesgo_crediticio python=3.9 
conda activate env_riesgo_crediticio
```
```bash
pip install -r requirements.txt
```

### 3. Opción alternativa — Docker

```bash
docker build -t riesgo-crediticio .
docker run -p 8501:8501 riesgo-crediticio
```

---

## Uso

### Ejecutar la aplicación de monitoreo

```bash
streamlit run model_monitoring.py
```

La aplicación quedará disponible en `http://localhost:8501`.

### Configurar rutas (antes de ejecutar)

En `model_monitoring.py`, actualiza las rutas al modelo y al dataset según tu entorno local:

```python
model_path = 'models/model.pkl'
data_path  = 'Base_de_datos.xlsx'
```

---

## App de Monitoreo

La aplicación está organizada en **cuatro pestañas principales** y un **menú lateral**:

### Predecir
Formulario interactivo para ingresar manualmente las características de un cliente y obtener la **probabilidad de pago** en tiempo real. Las variables de entrada incluyen:

- Salario, edad y plazo del crédito
- Cuota pactada, deuda total e ingreso disponible
- Ratio de endeudamiento y saldo total
- Cantidad de créditos vigentes por sector (financiero, cooperativo, real)
- Tipo laboral y tendencia de ingresos

### Cargar Datos
Permite subir un nuevo dataset en formato **CSV** o **Excel (.xlsx)** para evaluar el comportamiento del modelo sobre datos recientes.

### Evaluar Data Drift
Detecta cambios estadísticos entre el dataset de entrenamiento original y el nuevo dataset cargado. Ver sección [Data Drift](#data-drift).

### Visualización
Dashboard con las métricas de evaluación del modelo:

- **Curva Precision-Recall** — con Average Precision Score
- **Curva ROC** — con área bajo la curva (AUC)
- **Matriz de Confusión** — con clasificación por umbral de 0.5

---

## Data Drift

El tab **"Evaluar Data Drift"** compara estadísticamente el conjunto de entrenamiento original con el nuevo dataset cargado, usando dos pruebas según el tipo de variable:

### Variables Numéricas — Test de Kolmogorov-Smirnov (KS)

Compara la distribución de cada variable numérica entre ambos datasets. Un **p-value < 0.05** indica que la distribución cambió significativamente.

| Variable | KS Stat | P-Value | Drift |
|---|---|---|---|
| salario_cliente | 0.08 | 0.03 | ⚠️ Sí |
| edad_cliente | 0.04 | 0.42 | ✅ No |
| ... | ... | ... | ... |

### Variables Categóricas — Prueba Chi-Cuadrado (χ²)

Compara la frecuencia de cada categoría entre ambos datasets mediante una tabla de contingencia. Un **p-value < 0.05** indica drift en la distribución categórica.

| Variable | Chi2 Stat | P-Value | Drift |
|---|---|---|---|
| tipo_laboral | 1.23 | 0.27 | ✅ No |
| tendencia_ingresos | 8.45 | 0.01 | ⚠️ Sí |

> **Nota:** Los valores nulos son eliminados antes de aplicar las pruebas. Solo se evalúan las columnas presentes en ambos datasets.

---

## Variables del Modelo

Recuerda respetar estas variables al ingresar datos manualmente o al cargar un nuevo dataset, ya que el modelo fue entrenado con estas características específicas:

| Variable | Tipo | Descripción |
|---|---|---|
| `salario_cliente` | Numérica | Salario mensual del cliente |
| `edad_cliente` | Numérica | Edad en años |
| `plazo_meses` | Numérica | Plazo del crédito en meses |
| `cuota_pactada` | Numérica | Cuota mensual acordada |
| `deuda_total` | Numérica | Capital prestado + otros préstamos |
| `ingreso_disponible` | Numérica | Ingresos menos cuota pactada |
| `ratio_endeudamiento` | Numérica | Relación deuda / ingreso |
| `saldo_total` | Numérica | Saldo total del cliente |
| `cant_creditosvigentes` | Numérica | Número de créditos activos |
| `creditos_sectorFinanciero` | Numérica | Créditos en sector financiero |
| `creditos_sectorCooperativo` | Numérica | Créditos en sector cooperativo |
| `creditos_sectorReal` | Numérica | Créditos en sector real |
| `tipo_laboral` | Categórica | Empleado / Independiente |
| `tendencia_ingresos` | Categórica | Creciente / Decreciente / Estable |

**Variable objetivo:** `Pago_atiempo` — `1` si el cliente pagó a tiempo, `0` si entró en mora.

---

## Licencia

Este proyecto está bajo la licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

<p align="center">Desarrollado por <a href="https://github.com/JuanCaVa21">JuanCaVa21</a></p>
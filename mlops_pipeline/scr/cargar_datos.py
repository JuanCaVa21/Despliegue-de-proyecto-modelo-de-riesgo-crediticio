# Carga de datos de Excel

import pandas as pd

def cargar_datos():
    """
    Carga de los datos
    
    returns:
        df: Dataframe con los datos
    """

    ruta_excel = '/Users/juanv/Documents/GitHub/Despliegue-de-proyecto-modelo-de-riesgo-crediticio/data/Base_de_datos.xlsx'

    # En caso de que el archivo no exista
    if not ruta_excel:
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta_excel}")

    # Creamos nuestra variable a invocar despues
    df = pd.read_excel(ruta_excel) 

    return df

# Ejecutamos el codigo 
if __name__ == "__main__":
    try:
        df = cargar_datos()
        print(f"Se cargaron correctamente: {df.shape} filas y columnas")
    except Exception as e:
        print(f"Error al cargar: {e}")
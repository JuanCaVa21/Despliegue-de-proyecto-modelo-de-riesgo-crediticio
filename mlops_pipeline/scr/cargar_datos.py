############################
###### CARGA DE DATOS ######

# La funcion cargar_datos(): nos devuelve el dataframe
# que usaremos en los demas scripts

from pathlib import Path
import pandas as pd

def cargar_datos():
    """
    Carga de los datos
    
    cargar_datos(..):
        
        Carga los datos que le pasemos. Mediante la libreria Pathlib 
        podemos acceder y guardar la ruta como un __file__. 
        ruta_base: Empezamos desde la ruta que nos encontramos en el momento.
        ruta_excel: Es necesario isar parent la cantidad de veces igual a la cantidad de carpetas que se 
                    profundizo en nuestro caso solamente 2.
        
    df: Sera la variable que usaremos como dataframe en nuestro notebook
    """

    # Usamos la ubicación del script para encontrar el Excel
    ruta_base = Path(__file__).resolve().parent
    
    # Si el Excel está en la raíz del repo y el script en /scr, 
    # subimos dos niveles con .parent.parent
    ruta_excel = ruta_base.parent.parent / "Base_de_datos.xlsx"

    # En caso de que el archivo no exista
    if not ruta_excel.exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta_excel}")

    # Creamos nuestra variable a invocar despues
    df = pd.read_excel(ruta_excel) 

    return df

# Ejecutamos el codigo 
if __name__ == "__main__":
    try:
        df = cargar_datos()
        print(f"Se cargaron correctamente: {len(df)} filas")
    except Exception as e:
        print(f"Error al cargar: {e}")
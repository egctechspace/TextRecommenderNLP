# Imporarcion de dependencias --------------------------------------------------------
import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn import preprocessing
# -------------------------------------------------------------------------------------
class FileOpener:
    """
    Clase para abrir un archivo como un DataFrame de pandas.
    """
    def __init__(self, file_path, encoding = "utf-8"):
        """
        Inicializa la clase con la ruta relativa al archivo y la codificación del archivo.
        
        Argumentos:
            file_path (str): La ruta al archivo.
            encoding (str): La codificación del archivo (opcional). Por defecto es "utf-8".
        """
        self.file_path = file_path 
        self.encoding = encoding
        
        # Verifica si el archivo existe y lo abre como un DataFrame de pandas
        if os.path.exists(self.file_path):
            self.df = pd.read_csv(self.file_path, sep=",", decimal=",", encoding = encoding)
        else:
            self.df = None

    def get_df(self):
        """
        Devuelve el DataFrame si el archivo existe.
        
        Devuelve:
            pd.DataFrame/None: El DataFrame o None si el archivo no existe.
        """
        if self.df is None:
            print('Archivo no encontrado')
        return self.df
    
    
class ExploredData():
    """
    Clase para manipular y analizar datos en un DataFrame de pandas.
    Hereda de la clase FileOpener para abrir el archivo y obtener el DataFrame.
    """
    def __init__(self, df):
        """
        Inicializa la clase con el DataFrame a analizar.
        
        Argumentos:
            df (pd.DataFrame): El DataFrame a analizar.
        """
        self.df = df

    def isnull_values(self):
        """
        Devuelve una lista con el número de valores nulos en cada columna del DataFrame.
        
        Devuelve:
            list: Una lista con el número de valores nulos en cada columna.
        """
        if self.df is not None:
            return list(self.df.isnull().sum())

    def group_by_column(self, column):
        """
        Agrupa el DataFrame por la columna especificada y devuelve el tamaño de cada grupo.
        
        Argumentos:
            column (str): La columna por la que agrupar el DataFrame.
        
        Devuelve:
            pd.Series: Una serie con el tamaño de cada grupo.
        """
        if self.df is not None:
            return self.df.groupby(column).size()

    def unique_values_col(self, column):
        """
        Devuelve el número de valores únicos en la columna especificada del DataFrame.
        
        Argumentos:
            column (str): La columna del DataFrame.
        
        Devuelve:
            int: El número de valores únicos en la columna.
        """
        if self.df is not None:
            return len(self.group_by_column(column).keys())

    def count_var(self, column):
        """
        Devuelve el número de registros únicos de una variable en un DataFrame.
        
        Argumentos:
            column (str): El nombre de la columna.
        
        Devuelve:
            pd.DataFrame: Un DataFrame con los valores únicos y su conteo.
        """
        countClass = self.df[column].value_counts()
        countClassDf = pd.DataFrame({"class":countClass.index,
                                     "count":countClass.values})
        return countClassDf

    def info_DataSet(self):
        """
        Devuelve información básica (columnas, conteo, tipo, grupos) en formato pandas.DataFrame sobre el DataFrame original.
        
        Devuelve:
            pd.DataFrame: Un DataFrame con información sobre cada columna del DataFrame original.
        """
        if self.df is not None:
            null_values = list(self.df.isnull().sum())
            print(f"Forma datos \n - Filas: {self.df.shape[0]} \n - Columnas: {self.df.shape[1]}")
            infor = pd.DataFrame({"Conteo":self.df.count(), 
                                  "Tipo":self.df.dtypes, 
                                  "Nulos": self.isnull_values(),
                                  "Valores unicos": [self.unique_values_col(col) for col in self.df.columns]})
            # Valores perdidos - Mapa de calor  (Solo si existen)
            if sum(null_values) >0:
                sns.heatmap(self.df.isnull().astype(int), cbar=False)
                plt.show()
            return infor
    
    def drop_columns_by_name(self, columns):
        """
        Elimina una o varias columnas de un pandas.DataFrame si contiene(n) el texto indicado
        
        Argumentos: 
            columns (list): Lista con las columnas a elimimnar del df 
        """
         
         # Verifica si las columnas existen en el DataFrame y las elimina
        to_drop = [column for column in columns if column in self.df.columns] 
        not_drop = [column for column in columns if column not in self.df.columns]
        self.df = self.df.drop(to_drop, axis=1)

    def LabelEncoder_class(self, column_original, column_output=None):
        """
        Transforma las etiquetas de texto en una columna de un DataFrame en valores numéricos utilizando la clase LabelEncoder.
        
        Argumentos:
            column_original (str): La columna del DataFrame que contiene las etiquetas de texto a transformar.
            column_output (str): La columna del DataFrame donde se almacenarán los valores numéricos resultantes (opcional).
                                Si no se proporciona, se sobrescribirá la columna original.
        
        Devuelve:
            pd.DataFrame: El DataFrame con las etiquetas de texto transformadas en valores numéricos.
        """
        # Si no se especifica una columna de salida, se sobrescribe la columna original
        if column_output == None:
            column_output = column_original
        label_encoder = preprocessing.LabelEncoder()                 # Crear el objeto 
        label_encoder.fit(self.df[column_original])                           # Entrenamos 
        self.df[column_output] = label_encoder.transform(self.df[column_original])   # Transformamos
        return self.df

    
    
class Graphics: 
    
    def __init__(self, df):
        """
        Inicializa una instancia de la clase con el DataFrame a utilizar para crear los gráficos.
        
        Argumentos:
            df (pd.DataFrame): El DataFrame a utilizar para crear los gráficos.
        """
        self.df = df
        
    
    def plot_bar(self, X, Y, title, decimals, color_palette = None, color_var = None ):
        """
        Crea un gráfico de barras utilizando las columnas especificadas del DataFrame para el eje x y el eje y.
        
        Argumentos:
            X (str): La columna del DataFrame que se utilizará para el eje x del gráfico.
            Y (str): La columna del DataFrame que se utilizará para el eje y del gráfico.
            title (str): El título del gráfico.
            decimals (int): El número de decimales a mostrar en las etiquetas del gráfico.
            color_palette (list): La paleta de colores a utilizar para el gráfico (opcional).
            color_var (str): La columna del DataFrame que se utilizará para colorear las barras del gráfico (opcional).
        Devuelve:
            plotly.graph_objs.Figure: Una figura de Plotly con el gráfico de barras.
        """
        # Crear el gráfico de barras
        if color_palette is not None:     # Si esta definida, si no sin paleta
            fig = px.bar(self.df, x = X, y= Y, 
                        title = title,
                        text_auto = decimals,
                        color = color_var,
                        labels = {"class": X, "count": Y},
                        color_discrete_sequence=color_palette)
        else:
            fig = px.bar(self.df, x = X, y= Y, 
                        title = title,
                        text_auto = decimals,
                        color = color_var,
                        labels = {"class": X, "count": Y})
        return fig
    
    def plot_line(self, X, Y, title, color_palette): 
        """
        Crea un gráfico de líneas utilizando las columnas especificadas del DataFrame para el eje x y el eje y.
        
        Argumentos:
            X (str): La columna del DataFrame que se utilizará para el eje x del gráfico.
            Y (str): La columna del DataFrame que se utilizará para el eje y del gráfico.
            title (str): El título del gráfico.
            color_palette (list): La paleta de colores a utilizar para el gráfico.
        Devuelve:
            plotly.graph_objs.Figure: Una figura de Plotly con el gráfico de líneas.
        """
        fig = px.line(self.df, x = X, y = Y,
                    title = title,
                    labels = {"class": X,
                                "count": Y},
                    color = X,
                    color_discrete_sequence = color_palette
                    )
        return fig

# Imporarcion de dependencias --------------------------------------------------------
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import re
from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import collections
import emot
# -------------------------------------------------------------------------------------

class FileOpener:
    """
    Clase para abrir un archivo como un DataFrame de pandas.
    """
    def __init__(self, relative_path):
        """
        Inicializa la clase con la ruta relativa al archivo.
        :param relative_path: La ruta relativa al archivo.
        """
        self.relative_path = relative_path
        self.cwd = os.getcwd()
        self.file_path = os.path.join(self.cwd, self.relative_path)

        if os.path.exists(self.file_path):
            self.df = pd.read_csv(self.file_path, sep=",", decimal=",")
        else:
            self.df = None

    def get_df(self):
        """
        Devuelve el DataFrame si el archivo existe.
        :return: El DataFrame o None si el archivo no existe.
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
        Inicializa la clase.
        :param relative_path: La ruta relativa al archivo.
        """
        self.df = df

    def isnull_values(self):
        """
        Devuelve una lista con el número de valores nulos en cada columna del DataFrame.
        :return: Una lista con el número de valores nulos en cada columna.
        """
        if self.df is not None:
            return list(self.df.isnull().sum())

    def group_by_column(self, column):
        """
        Agrupa el DataFrame por la columna especificada y devuelve el tamaño de cada grupo.
        :param column: La columna por la que agrupar el DataFrame.
        :return: Una serie con el tamaño de cada grupo.
        """
        if self.df is not None:
            return self.df.groupby(column).size()

    def unique_values_col(self, column):
        """
        Devuelve el número de valores únicos en la columna especificada del DataFrame.
        :param column: La columna del DataFrame.
        :return: El número de valores únicos en la columna.
        """
        if self.df is not None:
            return len(self.group_by_column(column).keys())

    def count_var(self, column):
        """
        Devuelve el número de registros únicos de una variable.
        :param column: El nombre de la columna.
        """
        countClass = self.df[column].value_counts()
        countClassDf = pd.DataFrame({"class":countClass.index,
                                     "count":countClass.values})
        return countClassDf

    def info_DataSet(self):
        """
        Devuelve información básica (columnas, conteo, tipo, grupos) en formato pandas.DataFrame.
        :return: Un DataFrame con información sobre cada columna del DataFrame original.
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
        Elimina una columna de un pandas.DataFrame si contiene el texto indicado
        Argumentos: 
            df     - Set de datos. Debe ser un objeto tipo pandas.DataFrame
            column - Columnas a elimimnar del df 
        """
        to_drop = [column for column in columns if column in self.df.columns] 
        not_drop = [column for column in columns if column not in self.df.columns]
        df_ = self.df.drop(to_drop, axis=1)
        if len(not_drop) > 0: 
            print(f"No se han encontrado las siguientes variables: {' '.join(not_drop)}")
        return df_
    
    
    def LabelEncoder_class(self, column):
        """
        Transforma las etiquetas de texto a valores numericos
        Argumentos:
            df - Set de datos. Debe ser un objeto tipo pandas.DataFrame
            column - Columna a elimimnar del df  
        """
        label_encoder = preprocessing.LabelEncoder()                 # Crear el objeto 
        label_encoder.fit(self.df[column])                           # Entrenamos 
        self.df[column] = label_encoder.transform(self.df[column])   # Transformamos
        return self.df
    
    
class Grapher: 
    
    def __init__(self, df):
        """
        Inicializa la clase.
        :param relative_path: La ruta relativa al archivo.
        """
        self.df = df
        
    
    def plot_bar(self, X, Y, title, height, width, decimals, color=None):
        """
        Devuelve un grafico del número de registros unicos de una variable  
        Argumentos:
            df           - Set de datos. Debe ser un objeto tipo pandas.DataFrame   
            column       - Nombre de la columna 
            name_x_label - nombre del eje X
            title        - Titulo del grafico 
        """
        # Grafica de barras de clases
        fig = px.bar(self.df, x = X, y= Y, 
                     color = color,
                     title = title,  
                     height = height, width = width,
                     text_auto = decimals,
                     labels = {"class": X,
                               "count": Y})
        return fig
    
    def plot_line(self, X, Y, title, height, width, decimals): 
        """
        Genera un gráfico de líneas que muestra cómo cambia el número de ocurrencias de 
        cada sentimiento a lo largo del tiempo.
        :param column_plot: La columna del DataFrame que se utilizará para el eje y del gráfico.
        :return: Una figura de Plotly con el gráfico de líneas.
        """
        fig = px.line(self.df, x = X, y = Y,
                      color = X,
                      title = title,  
                      height = height, width = width, 
                      labels = {"class": X,
                                "count": Y})
        return fig
    


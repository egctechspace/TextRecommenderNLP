# Imporarcion de dependencias --------------------------------------------------------
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
# -------------------------------------------------------------------------------------

class UnsupervisedAnalysis():
    """
    Esta clase realiza un análisis de sentimiento no supervisado en un texto dado.
    """
    def __init__(self, text):
        """
        Inicializa una instancia de la clase con el texto a analizar.
        :param text: El texto a analizar.
        """
        self.text = text
     
    def get_polarity(self):
        """
        Utiliza la clase SentimentIntensityAnalyzer para calcular la polaridad del texto lematizado.
        La polaridad es una medida de cuán positivo, negativo o neutral es el texto.
        :return: Un diccionario con las puntuaciones de polaridad para cada categoría (positivo, negativo, neutral y compuesto).
        """
        sentiment_analyzer = SentimentIntensityAnalyzer()
        score = sentiment_analyzer.polarity_scores(self.lematize_text())
        return score
    
    def polarity_result(self):
        """
        Devuelve la puntuación compuesta de polaridad del texto calculada por el método get_polarity.
        :return: La puntuación compuesta de polaridad del texto.
        """
        return self.get_polarity()["compound"]
    
    def analyze_sentiment(self, score = None):
        """
        Analiza el sentimiento del texto en función de su puntuación compuesta de polaridad.
        Si la puntuación es mayor o igual a 0.05, el sentimiento se considera positivo.
        Si la puntuación es menor o igual a -0.05, el sentimiento se considera negativo.
        De lo contrario, el sentimiento se considera neutral.
        :param score: La puntuación compuesta de polaridad del texto (opcional).
                      Si no se proporciona, se calculará utilizando el método polarity_result.
        :return: Una cadena que indica el sentimiento del texto (positivo, negativo o neutral).
        """
        # Score como parametro o se calcula, opcional. 
        if score == None: 
            score = self.polarity_result()
        if score >= 0.05: 
            sentiment = "Positive"  # Positive
        elif score > -0.05:
            sentiment = "Neutral"
        else:
            sentiment = "Negative"
        return sentiment

    
    
class AnalysisResults():
    """
    Esta clase realiza un análisis de sentimiento en un DataFrame utilizando la clase UnsupervisedAnalysis.
    """
    def __init__(self, df, analizer_column, result_column = "Sentiment"): 
        """
        Inicializa una instancia de la clase con el DataFrame a analizar y las columnas relevantes.
        :param df: El DataFrame a analizar.
        :param analizer_column: La columna del DataFrame que contiene el texto a analizar.
        :param result_column: La columna del DataFrame donde se almacenarán los resultados del análisis (opcional).
                              Por defecto es "Sentiment".
        """
        self.df = df
        self.analizer_column = analizer_column
        self.result_column = result_column
        self._result_df = None
    
    def polarity_resume(self):
        """
        Calcula las puntuaciones de polaridad para cada fila del DataFrame utilizando la clase UnsupervisedAnalysis.
        :return: Un nuevo DataFrame con las puntuaciones de polaridad para cada fila.
        """
        polarity_df = self.df[self.analizer_column].apply(lambda row: pd.Series(UnsupervisedAnalysis(row).get_polarity()))
        return polarity_df
    
    def sentiment_resume(self):
        """
        Calcula el sentimiento para cada fila del DataFrame utilizando la clase UnsupervisedAnalysis.
        :return: Un nuevo DataFrame con el sentimiento para cada fila.
        """
        analyze_df = self.df[self.analizer_column].apply(lambda row: pd.Series({self.result_column : UnsupervisedAnalysis(row).analyze_sentiment()}))
        return analyze_df
    
    def results_df(self):
        """
        Devuelve un nuevo DataFrame que combina el DataFrame original con las puntuaciones de polaridad y el sentimiento para cada fila.
        :return: Un nuevo DataFrame con los resultados del análisis de sentimiento.
        """
        if self._result_df is None:
            results = pd.concat([self.df, self.polarity_resume(), self.sentiment_resume()], axis=1)
            self._result_df = results
        return self._result_df
    
    def plot_bar_sentiments(self):
        """
        Genera un gráfico de barras que muestra la distribución de los sentimientos en el DataFrame.
        :return: Una figura de Plotly con el gráfico de barras.
        """
        if self._result_df is None:
            self._result_df = self.results_df()
        fig = px.histogram(self._result_df, x = self.result_column, color = self.result_column)
        fig.update_layout(title_text = f"{self.result_column} analized")
        return fig
    
    def time_series_results(self, time_column = created_date,  time_freq = '10s'):
        """
        Agrupa los resultados del análisis de sentimiento por intervalos de tiempo y cuenta el número de ocurrencias de cada sentimiento en cada intervalo.
        :param time_column: La columna del DataFrame que contiene la información de tiempo (opcional).
                    Por defecto es created_date.
        :param time_freq: La frecuencia con la que se agruparán los resultados (opcional).
                    Por defecto es '10s' (cada 10 segundos).
        :return: Un nuevo DataFrame con los resultados agrupados por intervalos de tiempo y una cadena con el nombre de la columna de tiempo.
        """
        if self._result_df is None:
            self._result_df = self.results_df()
        # Agrupar cada x tiempo, la variable Sentiemientos y contar
        group_time_by_sentiments = self._result_df.groupby([pd.Grouper(key=time_column, freq= time_freq), self.result_column]).count()
        # Descomponer y ñadir 0 a las fechas sin valores
        add_missing_values = group_time_by_sentiments.unstack(fill_value=0)
        # Componemos y reseteamos indices
        resume = add_missing_values.stack().reset_index()
        #Formato de fecha
        resume[time_column] = pd.to_datetime(resume[time_column]).apply(lambda date: date.strftime('%y-%m-%d %H:%M:%S'))
        return resume, time_column

    def plot_time_series(self, column_plot): 
        """
        Genera un gráfico de líneas que muestra cómo cambia el número de ocurrencias de cada sentimiento a lo largo del tiempo.
        :param column_plot: La columna del DataFrame que se utilizará para el eje y del gráfico.
        :return: Una figura de Plotly con el gráfico de líneas.
        """
        time_series, time_column = self.time_series_results()
        fig = px.line(time_series, 
            x = time_column,
            y = column_plot,
            color = self.result_column)
        return fig

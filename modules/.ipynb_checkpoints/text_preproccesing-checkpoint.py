# Imporarcion de dependencias --------------------------------------------------------
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
# -------------------------------------------------------------------------------------

class TextPreProccessing():
    """
    Esta clase limpia y preprocesa texto para dejar solo el contenido más relevante.
    """
    def __init__(self, text, language= "english"):
        """
        Inicializa una instancia de la clase con el texto a preprocesar y el idioma del texto.
        :param text: El texto a preprocesar. Puede ser una cadena o una lista de cadenas.
        :param language: El idioma del texto (opcional). Por defecto es "english".
        """
        self.text = text
        self.language = language
    
    def join_text(self):
        """
        Si el texto es una lista de cadenas, las une en una sola cadena separada por espacios.
        :return: El texto como una única cadena.
        """
        # Si es un conjunto de textos se unen en uno solo
        if (type(self.text) != str):
            self.text = ' '.join(self.text)
        return self.text
            
    def clean_structure(self):
        """
        Limpia el formato del texto. Elimina todos los caracteres que no sean letras y convierte el texto a minúsculas.
        :return: El texto con formato limpio.
        """
        # Formato a todas las palabras
        self.text = re.sub('\W', ' ', self.text)   # Solo letras
        self.text = self.text.lower()              # En minúscula
        return self.text
    
    def stop_words(self):
        """
        Devuelve un conjunto de palabras comunes en el idioma especificado que deben excluirse del análisis.
        :return: Un conjunto de palabras comunes en el idioma especificado.
        """
        # Lista de palabras a excluir, ej: (the, this, a,...)
        stopwords_set = set(stopwords.words(self.language))
        return stopwords_set

    def tokenizate_text(self): 
        """
        Separa el texto en palabras individuales utilizando la función word_tokenize.
        :return: Una lista de tokens.
        """
        tokens = word_tokenize(self.text)
        return tokens
    
    def lematize_text(self):
        """
        Aplica la lematización a cada token en el texto utilizando la clase WordNetLemmatizer.
        La lematización es el proceso de reducir una palabra a su forma base o raíz.
        :return: El texto lematizado como una cadena.
        """
        lemmatizer = WordNetLemmatizer()
        lemmatizer_text = [lemmatizer.lemmatize(token, pos='v') for token in self.tokenizate_text()]
        self.text = ' '.join(lemmatizer_text)
        return self.text 
    
    def clean_text(self):
        """
        Limpia y preprocesa el texto. Aplica los métodos join_text, clean_structure y stop_words para dejar solo el contenido más relevante.
        :return: El texto limpio y preprocesado como una cadena.
        """
        # Aplicamos métodos anteriores
        self.join_text() 
        self.clean_structure() 
        self.lematize_text()
        stopwords_set = self.stop_words()
        # Filtrar palabras 
        cleaned_word = ([word for word in self.text.split()   # La palabra se añade si:
                             if word not in stopwords_set          # no es excluyente
                                 and 'http' not in word            # no es una url 
                                 and not word.startswith('@')      # no es mención @
                                 and not word.startswith('#')      # no es hashtag #
                                 and word != 'RT'                  # no es retweet RT
                                 and len(word)>2                   # si tiene más de 2 letras
                         ])
        cleaned_word = " ".join(cleaned_word)
        return cleaned_word
    
    def extract_emojis(self):
        """
        Extrae los emojis del texto utilizando la biblioteca emot.
        :return: Una lista con los emojis encontrados en el texto o None si no se encontraron emojis.
        """
        # Extrae emojis del texto
        emot_obj = emot.core.emot()         # Crea el objeto emot 
        res = emot_obj.emoji(self.text)     # Buscar el emoji
        if len(res['value'])>0:             # Si hay los devuelve: 
            return res['value']             # (evita listas vacías)
        
        
        
class DataExploration():
    """
    Esta clase realiza una exploración de datos en un DataFrame.
    """
    def __init__(self, df):
        """
        Inicializa una instancia de la clase con el DataFrame a explorar.
        :param df: El DataFrame a explorar.
        """
        self.df = df
    
    def join_text(self, column):
        """
        Une todas las cadenas en una columna del DataFrame en una sola cadena separada por espacios.
        :param column: La columna del DataFrame que contiene las cadenas a unir.
        :return: Una única cadena con todas las cadenas de la columna unidas.
        """
        # Toda la columna en un string
        return " ".join(self.df[column])
    
    def wordcloud_draw(self, column, background_color, mask = None):
        """
        Genera una nube de palabras para una columna del DataFrame utilizando la biblioteca WordCloud.
        :param column: La columna del DataFrame que contiene el texto para generar la nube de palabras.
        :param background_color: El color de fondo de la nube de palabras.
        :param mask: Una máscara para dar forma a la nube de palabras (opcional).
                     Por defecto es None (sin forma).
        :return: Un objeto WordCloud con la nube de palabras generada.
        """
        # Formar imagen para visualización de las palabras
        wordcloud = WordCloud(stopwords = STOPWORDS,                    # STOPWORDS, descarta palabras sin importancia (the, a, for, this,...)
                                background_color = background_color,
                                width = 2500,                           # Ancho de la gráfica
                                height = 2000,                          # Altura de la gráfica
                                mask = mask
                                ).generate(self.join_text(column))
        plt.figure(1,figsize = (6, 6))
        plt.imshow(wordcloud)
        plt.axis('off')
        return wordcloud
    
    def words_ranking(self, column, n):
        """
        Genera un ranking con las n palabras más comunes en una columna del DataFrame.
        :param column: La columna del DataFrame que contiene el texto para generar el ranking.
        :param n: El número de palabras a incluir en el ranking.
        :return: Un nuevo DataFrame con las n palabras más comunes y sus frecuencias.
        """

        # Crea ranking con las n palabras mas nombradas 
        tokenized_word = word_tokenize(self.join_text(column)) # Tokenización para crear una lista.
        fdist = FreqDist(tokenized_word)
        fd = pd.DataFrame(fdist.most_common(n), columns = ["Word","Frequency"]).drop([0]).reindex()
        return fd
        
    def emoji_ranking(self, column):
        """
        Genera un ranking con los emojis más comunes en una columna del DataFrame.
        :param column: La columna del DataFrame que contiene los emojis para generar el ranking.
        :return: Un nuevo DataFrame con los emojis más comunes y sus frecuencias.
        """
        # Contamos los emoticonos que se repiten 
        self.df[column].apply(lambda x: collections.Counter(x))
        combined_counts = sum(self.df[column].apply(lambda x: collections.Counter(x)), collections.Counter())
        emoji_dict = dict(combined_counts)
        sorted_emoji_dict = dict(sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True))

        # Creamos un df con los resultados de los emojis
        data = {"emoji":sorted_emoji_dict.keys(),
                "number": sorted_emoji_dict.values()}
        dfEmojis = pd.DataFrame(data)
        return dfEmojis
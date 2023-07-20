import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Tokenize:
    def __init__(self, X):
        """
        Inicializador de la clase Tokenizer.
        
        Argumentos:
            X (list): Una lista de textos para tokenizar.
        """
        self.X = X       # Variable independiente
        self.padded_X = None
        self.word_index = None
        self.tokenizer_object = None
    
    def get_padded_X(self):
        """
        Devuelve las secuencias de texto tokenizadas y rellenadas.
        
        Devuelve:
            numpy.ndarray: Un array que contiene las secuencias de texto tokenizadas y rellenadas.
        """
        return self.padded_X
    
    def get_word_index(self):
        """
        Devuelve el mapeo de palabras a índices en el vocabulario.
        
        Devuelve:
            dict: Un diccionario que mapea palabras a índices en el vocabulario.
        """
        return self.word_index
    
    def get_tokenizer_object(self):
        """
        Devuelve el tokenizador entrenado.
        
        Devuelve:
            Tokenizer: Un objeto Tokenizer entrenado en el texto proporcionado.
        """
        return self.tokenizer_object
    
    def keras_tokenizer(self):
        """
        Tokeniza y rellena las secuencias de texto utilizando el Tokenizer de Keras.
        
        Devuelve:
            Tokenizer: Un objeto Tokenizer entrenado en el texto proporcionado.
        """
        if self.padded_X is None: 
            # Tokenizer 
            self.tokenizer_object = Tokenizer()           # Creamos el objeto tokeniador
            self.tokenizer_object.fit_on_texts(self.X)         # Lo entrenamos con todo el texto  

            self.word_index = self.tokenizer_object.word_index   # Colculamos el index del tokenizador 
        return self.tokenizer_object

    def pad_sequence(self, text = None, PADDING_TYPE = 'post', TRUNC_TYPE = 'post'):
        """
        Convierte el texto en secuencias y luego las rellena para que todas tengan la misma longitud.
        
        Argumentos:
            text (list): Una lista opcional de textos para tokenizar y rellenar. Si no se proporciona,
                         se utiliza el texto proporcionado al inicializador de la clase.
            PADDING_TYPE (str): El tipo de relleno para aplicar a las secuencias (por defecto 'post').
            TRUNC_TYPE (str): El tipo de truncamiento para aplicar a las secuencias (por defecto 'post').
        
        Devuelve:
            numpy.ndarray: Un array que contiene las secuencias de texto tokenizadas y rellenadas.
        """
        
        if self.tokenizer_object is None:
            self.keras_tokenizer()
            
        if text is None:
            text_to_sequence = self.X
        else:
            text_to_sequence = text
            
        # Creamos las secuencias de las dos variables de texto 
        sequence_X = self.tokenizer_object.texts_to_sequences(text_to_sequence) 
        
        # Con Pad_sequences hacemos todas las secuencias de igual longitud. 
        self.padded_X = pad_sequences(sequence_X, padding = PADDING_TYPE, truncating = TRUNC_TYPE)
        
        return self.padded_X



class Embeddings:
    def __init__(self, GLOVE_EMBEDDING_PATH):
        """
        Construsctor de la clase Embeddings.
        
        Argumentos:
            GLOVE_EMBEDDING_PATH (str): Ruta al archivo .txt que contiene los vectores de incrustación pre-entrenados.
        """
        self.GLOVE_EMBEDDING_PATH = GLOVE_EMBEDDING_PATH
        self.embeddings_index = None
        self.embeddings = None
        self.embeddings_size = None

    def get_embeddings_size(self):
        """
        Devuelve el tamaño de los vectores de incrustación.
        
        Devuelve:
            int: El tamaño de los vectores de incrustación.
        """
        return self.embeddings_size
    
    def create_embeddings_index(self):
        """
        Crea un diccionario a partir de un archivo .txt que contiene vectores de incrustación pre-entrenados.
        Cada clave es una palabra y cada valor es el vector de incrustación correspondiente.
        
        Devuelve:
            dict: Un diccionario que mapea palabras a sus vectores de incrustación.
        """
        self.embeddings_index = {}
        with open(self.GLOVE_EMBEDDING_PATH, 'r', encoding="utf8") as file:
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_index[word] = vector
        return self.embeddings_index
    
    def calculate_embeddings(self):
        """
        Agrupa todos los vectores de incrustación en un solo array y calcula el tamaño de los vectores.
        
        Devuelve:
            tuple: Un array que contiene todos los vectores de incrustación y el tamaño de los vectores.
        """
        if self.embeddings_index is None: 
            self.create_embeddings_index()
        
        self.embeddings = np.stack(self.embeddings_index.values())
        self.embeddings_size = self.embeddings.shape[1]
        return self.embeddings, self.embeddings_size
    
    def embedding_matrix(self, word_index):
        """
        Genera una matriz de incrustación para un vocabulario específico.
        
        Argumentos:
            word_index (dict): Un diccionario que mapea palabras a índices en el vocabulario.
        
        Devuelve:
            numpy.ndarray: Una matriz de incrustación donde la i-ésima fila contiene el vector 
                           de incrustación para la palabra cuyo índice es i en el vocabulario.
        """
        
        if self.embeddings_size is None: 
            self.calculate_embeddings()
        
        vocabulary_size = len(word_index)
        embeddings_matrix = np.zeros((vocabulary_size + 1, self.embeddings_size))
        
        for word, i in word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embeddings_matrix[i] = embedding_vector
                
        return embeddings_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Tokenizer:
    def __init__(self, X):
        """
        Inicializador de la clase Tokenizer.
        Argumentos:
            X: una lista de textos para tokenizar.
        """
        self.X = X       # Variable independiente
        self.padded_X = None
        self.word_index = None
    
    def get_padded_X(self):
        """
        Devuelve las secuencias de texto tokenizadas y rellenadas.
        Devuelve:
            Un array que contiene las secuencias de texto tokenizadas y rellenadas.
        """
        return self.padded_X
    
    def get_word_index(self):
        """
        Devuelve el mapeo de palabras a índices en el vocabulario.
        Devuelve:
            Un diccionario que mapea palabras a índices en el vocabulario.
        """
        return self.word_index
    
    def keras_tokenizer(self, PADDING_TYPE = 'post', TRUNC_TYPE = 'post'):
        """
        Tokeniza y rellena las secuencias de texto utilizando el Tokenizer de Keras.
        Argumentos:
            PADDING_TYPE: el tipo de relleno para aplicar a las secuencias (por defecto 'post').
            TRUNC_TYPE: el tipo de truncamiento para aplicar a las secuencias (por defecto 'post').
        Devuelve:
            Un array que contiene las secuencias de texto tokenizadas y rellenadas y un diccionario que mapea palabras a índices en el vocabulario.
        """
        if self.padded_X is None: 
            # Tokenizer 
            tokenizer = Tokenizer()           # Creamos el objeto tokeniador
            tokenizer.fit_on_texts(self.X)         # Lo entrenamos con todo el texto  

            # Creamos las secuencias de las dos variables de texto 
            sequence_X = tokenizer.texts_to_sequences(self.X)  

            # Con Pad_sequences hacemos todas las secuencias de igual longitud. 
            self.padded_X = pad_sequences(sequence_X, padding = PADDING_TYPE, truncating = TRUNC_TYPE)

            self.word_index = tokenizer.word_index   # Colculamos el index del tokenizador 
            return self.padded_X, self.word_index



class Embeddings:
    def __init__(self, GLOVE_EMBEDDING):
        """
        Inicializador de la clase Embeddings.
        Argumentos:
            GLOVE_EMBEDDING: ruta al archivo .txt que contiene los vectores de incrustación pre-entrenados.
        """
        self.GLOVE_EMBEDDING = GLOVE_EMBEDDING
        self.embeddings_index = None
        self.embeddings = None
        self.embeddings_size = None
    
    def create_embeddings_index(self):
        """
        Crea un diccionario a partir de un archivo .txt que contiene vectores de incrustación pre-entrenados.
        Cada clave es una palabra y cada valor es el vector de incrustación correspondiente.
        Devuelve:
            Un diccionario que mapea palabras a sus vectores de incrustación.
        """
        self.embeddings_index = {}
        with open(self.GLOVE_EMBEDDING, 'r', encoding="utf8") as file:
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
            Un array que contiene todos los vectores de incrustación y el tamaño de los vectores.
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
            vocab_size: el tamaño del vocabulario.
            word_index: un diccionario que mapea palabras a índices en el vocabulario.
        Devuelve:
            Una matriz de incrustación donde la i-ésima fila contiene el vector de incrustación para la palabra cuyo índice es i en el vocabulario.
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

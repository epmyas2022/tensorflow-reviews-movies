from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

data = pd.read_csv('./data/Reviews.csv')

reviews = data['review_es'].tolist()
label = data['sentimiento'].tolist()

class TransformMatriz:
    def info():
        return(reviews,label)
    
    def tokenizer():
         tokenizer = Tokenizer(num_words=10000,oov_token="<OOV>")
         tokenizer.fit_on_texts(reviews)

         return tokenizer


    def sequences(tokenizer):
        #Tokenizer
       
        #convertir reseñas a secuencias de numeros
        sequences = tokenizer.texts_to_sequences(reviews)

        #tamaño maiximo de secuencia

        max_length = max([len(x) for x in sequences])

        #Rellenar secuencias para tenga una misma longitud

        padded = pad_sequences(sequences, maxlen=max_length, padding='post')

        return (padded, max_length)


    
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from utils.Transform import TransformMatriz


model = load_model('./model/model.h5')

#datos del entrenamiento



# Ejemplos de reseñas para hacer predicciones

test = [
     'Terrible servicio, no lo recomiendo',
     'Excelente servicio, muy recomendado',
     'Una de las peliculas mas aburridas que he visto',
     'La forma en que me trata es la mas horrible',
     'Amo la pelicula de principio a fin',
     'Esa persona es muy mala',
]


# Convertir las reseñas en secuencias de números


tokenizer = TransformMatriz.tokenizer()

test_sequences = tokenizer.texts_to_sequences(test)

(padded_train_data, max_length) = TransformMatriz.sequences(tokenizer)

test_padded_secuences = pad_sequences(test_sequences, maxlen=max_length, padding='post')

predictions = model.predict(test_padded_secuences)


# Resultados de las predicciones

for review, prediction in zip(test, predictions):
    if(prediction > 0.6):
        print(f'{review} =  POSITIVA')
    else:
        print(f'{review} =  NEGATIVA')
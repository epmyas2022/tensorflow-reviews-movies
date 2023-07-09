import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from utils.Transform import TransformMatriz


#descomponer la data valor = resultado

reviews, label = TransformMatriz.info()

#tokenizar datos 

tokenizer = TransformMatriz.tokenizer()

#convertir reseñas a secuencias de numeros

#Rellenar secuencias para tenga una misma longitud

padded, max_length = TransformMatriz.sequences(tokenizer)

# Dividir los datos en conjuntos de entrenamiento y prueba

label_encoder = LabelEncoder()

x_train, x_test, y_train, y_test = train_test_split(padded, label, test_size=0.2, random_state=42)

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


# Crear modelo

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# Compilar modelo

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Entrenar modelo

num_epochs = 20
history = model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test), verbose=2)

# ver resultados


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epocas")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, "accuracy")


# Ejemplo de reseña para hacer una predicción

new_review = ['Esa pelicula esta sobrevalorada, no me gusto']

# Convertir la reseña en secuencia de números

padding_type='post'

sample_sequences = tokenizer.texts_to_sequences(new_review)

pad_secuence_test = pad_sequences(sample_sequences, maxlen=max_length, padding=padding_type, truncating=padding_type)

# Hacer la predicción

predict = model.predict(pad_secuence_test)

print(predict)

# Guardar el modelo para usarlo mas tarde

model.save('./model/model.h5')

# Guardar peso del modelo

model.save_weights('./model/weights.h5')
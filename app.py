#!/usr/bin/env python
# Este archivo usa el encoding: utf-8
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os
from flask import send_from_directory
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import random
import json
import pickle
import numpy as np
import nltk
from pydub import AudioSegment
nltk.download('punkt')
import nltk
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from nltk.stem import SnowballStemmer
from flask import jsonify



app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Carga el modelo en una variable global
modelo = load_model("modelo.h5")
# Load the tokenizer and classes
palabras = pickle.load(open("palabras.pkl", "rb"))
clases = pickle.load(open("clases.pkl", "rb"))
@app.route('/')
def index():
    # Página principal
    return render_template('index.html')

@app.route('/about')
def about():
    # Página Por que este traductor
    return render_template('About.html')

@app.route('/porque')
def porque():
    # Página Por que este traductor
    return render_template('Porque.html')

stemmer = SnowballStemmer('spanish')
palabras_ignoradas = ["?","¿","!","¡"]
archivo_datos = open("intents.json", encoding='utf-8').read()
intenciones = json.loads(archivo_datos)

def bolsa_palabras(s, palabras):
    bolsa = [0 for _ in range(len(palabras))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words if word not in palabras_ignoradas]

    for se in s_words:
        for i, w in enumerate(palabras):
            if w == se:
                bolsa[i] = 1
            
    return np.array(bolsa)

ERROR_THRESHOLD = 0.8

@app.route('/predecir', methods=['POST'])
def predecir():

    # Obtener el mensaje del usuario
    mensaje = request.form.get('mensaje')
    # Imprimir el mensaje
    print(mensaje)
    # El resto del código aquí ...
    # Convertir el patrón de diálogo a su representación numérica
    resultados = modelo.predict(np.array([bolsa_palabras(mensaje, palabras)]))[0]
    resultados = [[i,r] for i,r in enumerate(resultados) if r>ERROR_THRESHOLD]
    resultados.sort(key=lambda x: x[1], reverse=True)
    if len(resultados)>0:
         # Recibir la intención con mayor probabilidad
        id_clase = resultados[0][0]  
        for intencion in intenciones['intents']:
                    if intencion['name'] == clases[id_clase]:
                        # Recibir una respuesta aleatoria de la intención
                        print(random.choice(intencion['responses'])) 
                        prediccion = random.choice(intencion['responses'])            
                        respuesta = {
                            'respuesta': prediccion
                             }
    else:
        respuesta = {
            'respuesta': 'lo siento, no puedo entenderte'

        }                   
    return jsonify(respuesta)


if __name__ == '__main__':
    app.run(debug=False, threaded=False)

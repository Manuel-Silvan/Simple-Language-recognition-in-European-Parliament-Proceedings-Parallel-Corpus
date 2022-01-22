# -*- coding: utf-8 -*-

import numpy as np
import os
import re
import random as rd

import nltk
import pickle #Para guardar archivos


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


from tfidf import *
from bigrama import *

from scipy import sparse
lista = os.listdir()
r = re.compile("^europarl-v7(?!.*\.en$).*$")#

ficheros = list(filter(r.findall, lista))
idiomas = [
'Aleman',
'Español',
'Frances',
'Italiano',
'Polaco',
'Ingles'
]
ficheros.append('europarl-v7.es-en.en')
corpus_train = [[] for i in range(len(idiomas))]
corpus_val = [[] for i in range(len(idiomas))]


porcentaje_extraccion = 0.01
frases_entrenamiento = int(110000 * porcentaje_extraccion)
frases_test = int(10000 * porcentaje_extraccion)



for idx,val in enumerate(ficheros):
    with open(val, encoding="utf8") as var:
        texto = var.readlines()



        for i in range(frases_test):
            al = rd.randint(0,10)

            corpus_val[idx].append( texto[al+ 11*i] )
            corpus_train[idx] = corpus_train[idx] + texto[0 + 11*i:al + 11*i] + texto[al+1 + 11*i :11 + 11*i]



#Normalización
def normaliza(corpus_train,corpus_val):
    #Todo a minuscula

    corpus_train = [[j.lower() for j in i] for i in corpus_train]
    corpus_val = [[j.lower() for j in i] for i in corpus_val]




    #Todos los simbolos que no sean apostrofes no son utiles
    #Un apostrofe que aparezca solo es inutil, pero si aparece entre palabras no
    #Numeros en general no dan significado, si siquiera aun estando en una palabra
    corpus_train = [[re.sub(r'(\s[\']\s)|([0-9])|[!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]', '', j) for j in i] for i in corpus_train]
    corpus_val = [[re.sub(r'(\s[\']\s)|([0-9])|[!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]', '', j) for j in i] for i in corpus_val]

    return  corpus_train, corpus_val


corpus_train, corpus_val = normaliza(corpus_train, corpus_val)
#100 palabras más frecuentes de cada idioma



freq100 = [nltk.FreqDist(nltk.word_tokenize(' '.join(i))).most_common(100) for i in corpus_train]
word100 = [[j[0] for j in i] for i in freq100]
#Normalizacion y extracción de características TF-IDF



vectorizer = tfidf(20)

#Concatenamos las frases para dejar una matriz con los idiomas

corpus_train_unido = [' '.join(i) for i in corpus_train]

TF_IDF = vectorizer.fit_transform(corpus_train_unido)

#Guardamos la matriz


with open('TF_IDF', 'wb') as f:
    pickle.dump(TF_IDF, f)
with open('vectorizer', 'wb') as f:
    pickle.dump(vectorizer, f)
#Cargamos si necesario
with open('TF_IDF','rb') as f:
    TF_IDF = pickle.load(f)
with open('vectorizer','rb') as f:
    vectorizer = pickle.load(f)

#MODELO SIMPLE DE CLASIFICACION





#Vectorizamos con el modelo anterior un documento, y hacemos el producto escalar con la columna idioma de la matriz tf_idf, el que de un producto escalar
# mas alto gana pues sera el que este más cerca
def clasifica(modelo, TF_IDF,X_test):
    features = modelo.transform(X_test)
    resultado = np.dot(np.array(TF_IDF),np.array(features).T)
    h = np.argmax(resultado, axis = 0)
    h = np.squeeze(np.asarray(h))
    return h

corpus_val_unido = np.concatenate(corpus_val)

y_pred = clasifica(vectorizer,TF_IDF,corpus_val_unido)




y_test = np.ones(frases_test*len(idiomas))
for i in range(len(idiomas)):
    y_test[i*frases_test:(i+1)*frases_test] = i

#Analisis del rendimiento del modelo

print('Matriz de confusion')
print(confusion_matrix(y_test, y_pred))
# Accuracy
print('Accuracy: ',accuracy_score(y_test, y_pred))
# Recall

print('Recall (por idioma): ',recall_score(y_test, y_pred, average=None))
# Precision
print('Precision (por idioma): ',precision_score(y_test, y_pred, average=None))
#Limpieza de datos

#Recargamos todo pero filtrando antes frases inutiles

corpus_train = [[] for i in range(len(idiomas))]
corpus_val = [[] for i in range(len(idiomas))]

for idx,val in enumerate(ficheros):
    with open(val, encoding="utf8") as var:
        texto = var.readlines()
        #Lo nuevo

        Excluidos = list(np.concatenate(word100[:idx] + word100[idx+1:]))

        aux = []
        i = 0
        while(len(aux) < frases_entrenamiento):
            if(nltk.word_tokenize(texto[i]) not in Excluidos):
                aux.append(texto[i])

            i+=1
        texto = aux





        for i in range(frases_test):
            al = rd.randint(0,10)
            corpus_val[idx].append( texto[al+ 11*i] )
            corpus_train[idx] = corpus_train[idx] + texto[0 + 11*i:al + 11*i] + texto[al+1 + 11*i :11 + 11*i]



corpus_train, corpus_val = normaliza(corpus_train, corpus_val)
#APARTADO 5 TF-IDF CORPUS LIMPIO



vectorizer = tfidf(20)

corpus_train_unido = [' '.join(i) for i in corpus_train]
TF_IDF = vectorizer.fit_transform(corpus_train_unido)
corpus_val_unido = np.concatenate(corpus_val)
y_pred = clasifica(vectorizer,TF_IDF,corpus_val_unido)
#El y-test usamos el previo puesto que no ha cambiado

print('Matriz de confusion')

print(confusion_matrix(y_test, y_pred))
# Accuracy

print('Accuracy: ',accuracy_score(y_test, y_pred))
# Recall


print('Recall (por idioma): ',recall_score(y_test, y_pred, average=None))
# Precision

print('Precision (por idioma): ',precision_score(y_test, y_pred, average=None))

#APARTADO 6 N-GRAMAS




clf = bigrama(2)
clf.entrena(corpus_train)
corpus_val_unido = np.concatenate(corpus_val)
y_pred = clf.clasifica(corpus_val_unido)
y_test = np.ones(frases_test*len(idiomas))
for i in range(len(idiomas)):
    y_test[i*frases_test:(i+1)*frases_test] = i


print('Matriz de confusion')

print(confusion_matrix(y_test, y_pred))
# Accuracy

print('Accuracy: ',accuracy_score(y_test, y_pred))
# Recall


print('Recall (por idioma): ',recall_score(y_test, y_pred, average=None))
# Precision

print('Precision (por idioma): ',precision_score(y_test, y_pred, average=None))


#Apartado 7
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(min_df = 20)

corpus_train_unido = np.concatenate(corpus_train)
corpus_val_unido = np.concatenate(corpus_val)

X_train= vec.fit_transform(corpus_train_unido)
X_test = vec.transform(corpus_val_unido)
y_test = np.ones(frases_test*len(idiomas))
y_train = np.ones((frases_entrenamiento-frases_test)*len(idiomas))
for i in range(len(idiomas)):
    y_test[i*frases_test:(i+1)*frases_test] = i
    y_train[i*(frases_entrenamiento-frases_test):(i+1)*(frases_entrenamiento-frases_test)] = i

from sklearn.tree import DecisionTreeClassifier

model  = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('Matriz de confusion')

print(confusion_matrix(y_test, y_pred))
# Accuracy

print('Accuracy: ',accuracy_score(y_test, y_pred))
# Recall


print('Recall (por idioma): ',recall_score(y_test, y_pred, average=None))
# Precision

print('Precision (por idioma): ',precision_score(y_test, y_pred, average=None))

#Podemos saber que tokens han sido los más determinantes
n_tok = 10
index = np.argsort(model.feature_importances_)
tok = [index[-i-1] for i in range(n_tok)]
important = [vec.get_feature_names()[i] for i in tok]
print('Palabras más determinantes')
print(important)

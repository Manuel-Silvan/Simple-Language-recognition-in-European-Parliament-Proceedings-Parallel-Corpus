# -*- coding: utf-8 -*-

import numpy as np
import os
import re
import random as rd
import getopt, sys


import nltk
#nltk.download('punkt')
import pickle #Para guardar archivos


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


from tfidf import *
from bigrama import *

from scipy import sparse
lista = os.listdir()
r = re.compile("^europarl-v7.*$")#"^europarl-v7(?!.*\.en$).*$"

ficheros = list(filter(r.findall, lista))



normalize = 0
model = 0
# Remove 1st argument from the
# list of command line arguments
argumentList = sys.argv[1:]
 
# Options
options = "nm:"
 
# Long options
long_options = ["normalize", "model ="]
 
try:
    # Parsing argument
    arguments, values = getopt.getopt(argumentList, options, long_options)
     
    # checking each argument
    for currentArgument, currentValue in arguments:
 
        if currentArgument in ("-n", "--normalize"):
            normalize = 1
             
        
             
        elif currentArgument in ("-m", "--model"):
            model = int(currentValue)
             
except getopt.error as err:
    # output error, and return with an error code
    print (str(err))



idiomas = [
'Aleman',
'Español',
'Frances',
'Italiano',
'Polaco',
'Ingles'
]
idiomas = idiomas.sort()


corpus_train = [[] for i in range(len(ficheros))]
corpus_val = [[] for i in range(len(ficheros))]


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

if(normalize):
    #filtrar frases inutiles

    corpus_train = [[] for i in range(len(ficheros))]
    corpus_val = [[] for i in range(len(ficheros))]

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


#Normalizacion y extracción de características TF-IDF

if(model == 0): #TF-IDF scratch
    vectorizer = tfidf(20)
    #Concatenamos las frases para dejar una matriz con los idiomas
    corpus_train_unido = [' '.join(i) for i in corpus_train]
    TF_IDF = vectorizer.fit_transform(corpus_train_unido)

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
    y_test = np.ones(frases_test*len(ficheros))
    for i in range(len(ficheros)):
        y_test[i*frases_test:(i+1)*frases_test] = i


elif(model == 1):#bigrama scratch
    clf = bigrama(2)
    clf.entrena(corpus_train)
    corpus_val_unido = np.concatenate(corpus_val)
    y_pred = clf.clasifica(corpus_val_unido)
    y_test = np.ones(frases_test*len(ficheros))
    for i in range(len(ficheros)):
        y_test[i*frases_test:(i+1)*frases_test] = i
elif(model == 2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(min_df = 20)

    corpus_train_unido = np.concatenate(corpus_train)
    corpus_val_unido = np.concatenate(corpus_val)

    X_train= vec.fit_transform(corpus_train_unido)
    X_test = vec.transform(corpus_val_unido)
    y_test = np.ones(frases_test*len(ficheros))
    y_train = np.ones((frases_entrenamiento-frases_test)*len(ficheros))
    for i in range(len(ficheros)):
        y_test[i*frases_test:(i+1)*frases_test] = i
        y_train[i*(frases_entrenamiento-frases_test):(i+1)*(frases_entrenamiento-frases_test)] = i

    from sklearn.tree import DecisionTreeClassifier

    tree  = DecisionTreeClassifier()
    tree.fit(X_train,y_train)
    y_pred = tree.predict(X_test)


    #Podemos saber que tokens han sido los más determinantes
    print('********PALABRAS MAS RELEVANTES PARA DIFERENCIAR ENTRE IDIOMAS********')
    n_tok = 10
    index = np.argsort(tree.feature_importances_)
    tok = [index[-i-1] for i in range(n_tok)]
    important = [vec.get_feature_names()[i] for i in tok]

    print(important)



else:
    print('ERROR')



#Analisis del rendimiento del modelo
modelos = ["TF-IDF from-scratch", "bigram from-scratch","sklearn TF-IDF"]
norm = ["sin limpiar corpus","corpus limpio"]
print("********Rendimiento "+ modelos[model]+" "+norm[normalize]+" ********")
print('Matriz de confusion')
print(confusion_matrix(y_test, y_pred))
# Accuracy
print('Accuracy: ',accuracy_score(y_test, y_pred))
# Recall

print('Recall (por idioma): ',recall_score(y_test, y_pred, average=None))
# Precision
print('Precision (por idioma): ',precision_score(y_test, y_pred, average=None))





















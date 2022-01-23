import numpy as np
import nltk

from scipy import sparse

class bigrama:
    def __init__(self,min_df = 0):
        self.min_df = min_df
    def entrena(self, corpus):
        self.frecuencias = []
        self.bigrama = []
        self.vocabulario = []
        self.tam_vocab = []
        self.total_idioma = []
        self.n_idiomas =len(corpus)
        for i in range(self.n_idiomas):
            corpus_unido = nltk.word_tokenize(' '.join(corpus[i]))

            frecuencias = nltk.FreqDist(corpus_unido)
            vocab = [idx for idx,val in frecuencias.items() if val > self.min_df]#Pongo un umbral para que la tabla no sea tan grande

            if(self.min_df > 0):
                for j in range(len(corpus_unido)):
                    if(corpus_unido[j] not in vocab):
                        corpus_unido[j] = '<UNK>'


            vocab = np.unique(corpus_unido)
            tam_vocab = len(vocab)

            freqD = np.zeros(tam_vocab)
            bigr = sparse.csr_matrix((tam_vocab,tam_vocab))
            dict_vocab = dict(zip(vocab, list(range(tam_vocab))))
            for j in range(len(corpus_unido)):
                freqD[dict_vocab[corpus_unido[j]]] +=1


                if(j != 0 and corpus_unido[j] != '<UNK>'):
                    bigr[dict_vocab[corpus_unido[j]],dict_vocab[corpus_unido[j-1]]]+= 1
            self.frecuencias.append(freqD)
            self.bigrama.append(bigr)
            self.vocabulario.append(dict_vocab)
            self.total_idioma.append(len(corpus_unido))
            self.tam_vocab.append(tam_vocab)
    def clasifica(self, frases):


        corpus_tok =[nltk.word_tokenize(frases[i]) for i in range(len(frases))]



        p = np.zeros((self.n_idiomas,len(frases)))
        print(np.shape(p))
        for i in range(self.n_idiomas):
            for j in range(len(corpus_tok)):
                for k in range(len(corpus_tok[j])):
                    aux = corpus_tok[j][k]
                    aux2 = corpus_tok[j][k-1]
                    if(corpus_tok[j][k] not in list(self.vocabulario[i].keys())):
                        corpus_tok[j][k] = '<UNK>'
                    if(corpus_tok[j][k-1] not in list(self.vocabulario[i].keys())):
                        corpus_tok[j][k-1] = '<UNK>'
#
                    if(k == 0):
                        prob = (self.frecuencias[i][self.vocabulario[i][corpus_tok[j][k]]] +1) / (self.tam_vocab[i]+self.total_idioma[i])
                    else:
                        prob = (self.bigrama[i][self.vocabulario[i][corpus_tok[j][k]],self.vocabulario[i][corpus_tok[j][k-1]]] + 1)/(self.frecuencias[i][self.vocabulario[i][corpus_tok[j][k-1]]] + self.tam_vocab[i])

                    corpus_tok[j][k] =aux
                    corpus_tok[j][k-1]= aux2

                    p[i][j]+= np.log(prob)


        y_pred = [np.argmax(i) for i in p.T]
        return y_pred

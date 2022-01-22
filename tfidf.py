class tfidf:
    def __init__(self, min_df = 0):
        self.min_df = min_df
    def fit_transform(self,corpus):
        frecuencias = nltk.FreqDist(nltk.word_tokenize(' '.join(corpus)))
        vocabulario = [idx for idx,val in frecuencias.items() if val > self.min_df]#Pongo un umbral para que la tabla no sea tan grande
        self.vocabulario  = dict(zip(vocabulario, list(range(len(vocabulario)))))
        return self.transform(corpus)
    def transform(self,corpus):
        #Lo mismo pero asumiendo que el vocabulario es el previamente dado



        tf = self.term_frequency(corpus)
        idf = self.inverse_document_frequency(corpus)



        tabla = [[tf[i][j] * np.log(idf[j]) for j in range(len(list(self.vocabulario.keys())))] for i in range(len(corpus))]
        return tabla
    def term_frequency(self, corpus):
        longitud_doc = [len(i.split()) for i in corpus]
        corpus_tokenizado = [nltk.word_tokenize(i) for i in corpus]
        count_vector = [nltk.FreqDist(i )for i in corpus_tokenizado]


        tf = [[ count_vector[i][list(self.vocabulario.keys())[j]] / max(longitud_doc[i],1) for j in range(len(list(self.vocabulario.keys())))] for i in range(len(corpus))]
        return tf
    def inverse_document_frequency(self,corpus):
        n_doc = len(corpus)


        idf = [n_doc  /  np.max([1,sum([int(list(self.vocabulario.keys())[i] in j) for j in corpus]) ]) for i in range(len(list(self.vocabulario.keys())))]
        return idf

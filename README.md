# Simple-Language-recognition-in-European-Parliament-Proceedings-Parallel-Corpus
  The European Parliament Proceedings Parallel Corpus (1996-2011) (https://www.statmt.org/europarl/) is a well-known dataset in Natural Language Processing tasks, 
  it contains proceedings of the European Parliament in 21 European languages.  In this project we will only extract data from 6 languages 
  (German, French, Spanish, Italian, Polish and English), we will extract, preprocess, clean and normalize the data and after that we will train on that data 
  some quite simple classifiers that will be able to tell in which language a sentence is written.  
  
  This was originally an small project i did on university, and now I'm trying to formalize it so many other models or techniques can be tested in it.
  
  To test it, put any of the files that can be found in https://www.statmt.org/europarl/ in the folder that the Python files are in, these files contain a bunch of sentences in a certain
  language, the program works so that each file is a different language. 
  There are three simple classification models set:
  
  - TF-IDF programmed from-scratch + dot product
  - Bigram 
  - TF-IDF + DecissionTree both from sklearn
  
  The code lets you execute it with 2 parameters, add "-n" or "--normalize" to remove the 100 most frecuent word from the corpus and "-m" or "--model" to choose the technique to clasify.
  
  
  
  

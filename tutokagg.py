#!/usr/bin/env python
#  
# Pruebas para el Algoritmo 1.
#
# python tutokagg.py
# *************************************** #

import pandas as pd   
import numpy as np
from bs4 import BeautifulSoup  
import re
import nltk
from nltk.corpus import stopwords 

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))

    
train = pd.read_csv("labeledTrainData.tsv", header = 0, \
                    delimiter = "\t", quoting = 3)

print "Cleaning and parsing the training set movie reviews...\n"
clean_train_reviews_pos = []
clean_train_reviews_neg = []
num_reviews = train["review"].size
for i in xrange( 0, num_reviews ):
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, num_reviews )   
    if ((train["sentiment"][i]) == 1):
		clean_train_reviews_pos.append( review_to_words( train["review"][i] ))
    if ((train["sentiment"][i]) == 0):
		clean_train_reviews_neg.append( review_to_words( train["review"][i] ))

# # # # Implementando el Algoritmo 1 de Pancho & Pablo

# # # # Paso 1:
# Se van a usar dos Bags diferentes: 
# Bag negativa: conteniendo palabras que provengan de reviews negativas.
# Bag positiva: conteniendo palabras que provengan de reviews positivas.

# Por cada review: se ve que puntaje tiene la review. Si el puntaje es 
# positivo, sus palabras son introducidas en el Bag positivo
# En caso de que el puntaje sea negativo se introducen las palabras
# del review en el Bag negativo.

print "Creando una bag positiva y otra negativa...\n"
from sklearn.feature_extraction.text import CountVectorizer

vectorizer_pos = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

vectorizer_neg = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features_pos = vectorizer_pos.fit_transform(clean_train_reviews_pos)
train_data_features_neg = vectorizer_neg.fit_transform(clean_train_reviews_neg)
# Convierto el resultado en un numpy array
train_data_features_pos = train_data_features_pos.toarray()
train_data_features_neg = train_data_features_neg.toarray()

print train_data_features_pos.shape
print train_data_features_neg.shape

# Para ver las palabras
vocab_pos = vectorizer_pos.get_feature_names()
vocab_neg = vectorizer_neg.get_feature_names()

# # # # Paso 2:
# Se eliminaran las palabras que tengan una frecuencia muy baja.
# Luego mirando los dos Bags, si se encuentran palabras que tienen
# frecuencias similares en ambas Bags, se eliminan de ambas.
# Si una palabra aparece en ambas Bags, pero en una su frecuencia es
# mayor que en la otra, se elimina la palabra de la Bag que tiene menor frecuencia.

# Sumar las veces que aparece cada word
dist_pos = np.sum(train_data_features_pos, axis = 0)
dist_neg = np.sum(train_data_features_neg, axis = 0)

print dist_pos
print dist_neg

# Imprime la palabra y el numero de veces que aparece en total 
for count, tag in sorted([(count, tag) for tag, count in zip(vocab_pos, dist_pos)], reverse=False)[1:10]:
    print(count, tag)

print "\n\n"

for count, tag in sorted([(count, tag) for tag, count in zip(vocab_neg, dist_neg)], reverse=False)[1:10]:
    print(count, tag)

print "\n\n"


"""# Si una bag tiene frecuencias muy menores a otra, se eliminan de esa bag
# Es decir seteo la frecuencia en cero
for count, tag in zip(vocab_pos, dist_pos):
	for countneg, tagneg in zip(vocab_neg, dist_neg):
		if (count < countneg):
			tag = 0

print dist_pos

# Saco las frecuencias que sean menores a 100, solo por tirar un numero
palabras_pos = []
for (tag, count) in zip(vocab_pos, dist_pos):
	if (count > 100):
		palabras_pos.append(tag)

palabras_neg = []
for (tag, count) in zip(vocab_neg, dist_neg):
	if (count > 100):
		palabras_neg.append(tag)
"""

# # # # Paso 3:
# Finalmente se agarran las reviews sin resultado, se eliminan las
# stop words, exclamaciones y mayusculas. y se revisa si sus palabras
# pertenecen mayormente a la Bag con reviews positivas o a la Bag con reviews negativas.

# En este caso voy a probar con las reviews de labeledTrainData para ver
# si realmente funciona nuestra idea
print "El sentiment de esta review es %d (si es 1 es positivo)" % train["sentiment"][0]

clean_reviews = review_to_words( train["review"][0] )
clean_reviews = clean_reviews.split()

for word in clean_reviews:
	print word
	if word in vocab_pos:
		print "Esta"
	else:	
		print "No ta"









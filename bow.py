#!/usr/bin/env python
# 
# Pruebas para un Bag Of Words. Algoritmo 1.
# 
# python bow.py
# *************************************** #

import pandas as pd    
import numpy as np   

train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)

# # # # Empiezo a procesar el texto.
# Pruebo con una review por el momento
from bs4 import BeautifulSoup                
example1 = BeautifulSoup(train["review"][0])  

print "Review original:\n\n %s \n\n" % train["review"][0]
print "Review procesada:\n\n %s \n\n" % example1.get_text()

# Solo dejo letras, nada de numeros ni otros caracteres
import re
letters_only = re.sub("[^a-zA-Z]",           # Todo lo que no sean letras
                      " ",                   # Lo reemplazo con espacio
                      example1.get_text() )  # En este texto, example1
print "Review sin puntuacion ni numeros:\n\n %s \n\n" % letters_only

lower_case = letters_only.lower()        
words = lower_case.split()               
print "Listado de palabras, words, queda:\n\n %s \n\n" % words

import nltk
from nltk.corpus import stopwords 

words = [w for w in words if not w in stopwords.words("english")]
print "Sin stop words esto queda como:\n\n %s \n\n" % words

# Junto todas las palabras en un unico string separado por space 
textofinal = ( " ".join(words) )   

print "El texto final seria:\n\n %s \n\n" % textofinal

# # # # Implementando un Bag of Words decente
print "\n\n# # # # Pruebo el Bag of Words decente \n\n"

class Bag_of_words:
	# Un bow seria un diccionario que guarda word:elemarray
	# siendo elemarray la posicion en donde se guardara la frecuencia
	# en el feature vector generado en este proceso
	def __init__(self):
		self.bag = {}   
		self.featurevec = []
		self.contador = 0

	# Agrega una nueva key al bag, mismo caso que bow simple
	def agregar(self, key):
		if key in self.bag:
			self.featurevec[ self.bag[key] ] += 1
			self.contador += 1
		else:
			self.bag[key] = self.contador
			self.featurevec.append(1)
			self.contador += 1
			
	# Devuelve la frecuencia con la que aparece una key
	def frecuencia(self, key):
		if key in self.bag:
			return self.featurevec[ self.bag[key] ]
		else:
			return 0
			
	# Devuelve todas las keys que se guardan en el bag
	def palabrasGuardadas(self):
		return self.bag.keys()
		
	# Devuelve el feature vector con las frecuencias
	def featureVector(self):
		return self.featurevec
	
	# Devuelve el contador, solo para probar. ELIMINAR ESTO
	def contadorEn(self):
		return self.contador

bagPrueba = Bag_of_words()
print "Agrego la palabra perro a la bag\n"
bagPrueba.agregar('perro')
print "La palabra perro deberia ahora aparecer una vez. Su frecuencia es %d" % bagPrueba.frecuencia('perro')
print "La lista de palabras guardadas es %s \n" % bagPrueba.palabrasGuardadas()
print "Agrego mas palabras: gato, raton, perro\n"
bagPrueba.agregar('gato')
bagPrueba.agregar('raton')
bagPrueba.agregar('perro')
print "La palabra perro deberia tener frec 2. Su frecuencia es %d" % bagPrueba.frecuencia('perro')
print "La lista de palabras guardadas es %s \n" % bagPrueba.palabrasGuardadas()
print "La palabra elefante deberia tener frec 0. Su frecuencia es %d" % bagPrueba.frecuencia('elefante')
print "Contador deberia estar en 4, esta en", bagPrueba.contadorEn()
bagPrueba.agregar('raton')
print "La palabra raton deberia tener frec 2. Su frecuencia es %d" % bagPrueba.frecuencia('raton')
print "Feature vector deberia ser [perro, gato, raton], osea [2, 1, 2], y es:\n", bagPrueba.featureVector()


# # # # Implementando el Algoritmo 1 de Pancho & Pablo
# Paso 1:
# Se van a usar dos Bags diferentes: 
# Bag negativa: Bag conteniendo las palabras que provengan de reviews negativas.
# Bag positiva: Bag conteniendo las palabras que provengan de reviews positivas.

# Primero se toma el archivo que contiene las reviews con el resultado y
# por cada review, se eliminan las stop words, exclamaciones y mayusculas.
# Luego por cada review se ve que puntaje tiene la review. Si el puntaje
# es positivo, sus palabras son introducidas en el Bag positivo, agregandola
# cada palabra, si no estaba en el Bag o sumandole uno a la frecuencia.
# En caso de que el puntaje sea negativo se introducen las palabras
# del review en el Bag negativo.
"""
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

    
train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)

print "Cleaning and parsing the training set movie reviews...\n"
clean_train_reviews_pos = []
clean_train_reviews_neg = []
num_reviews = train["review"].size
for i in xrange( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, num_reviews )   
    if ((train["sentiment"][i]) == 1):
		clean_train_reviews_pos.append( review_to_words( train["review"][i] ))
    if ((train["sentiment"][i]) == 0):
		clean_train_reviews_neg.append( review_to_words( train["review"][i] ))
"""

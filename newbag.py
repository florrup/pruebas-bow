#!/usr/bin/env python
# 
# Pruebas para un Bag Of Words. Algoritmo 1.
# 
# python bow.py
# *************************************** #

import pandas as pd   
import numpy as np
from bs4 import BeautifulSoup  
import re
import nltk
from nltk.corpus import stopwords 


"""
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
"""
# # # # Implementando un Bag of Words decente
print "\n\n# # # # Pruebo el Bag of Words decente \n\n"

class Bag_of_words:
	# Un bow seria un diccionario que guarda word:elemarray
	# siendo elemarray la posicion en donde se guardara la frecuencia
	# en el feature vector generado en este proceso
	def __init__(self):
		self.bag = {}   
		self.featurevec_pos = []
		self.words = []
		self.featurevec_neg = [] 
		self.contador = 0

	# Agrega una nueva key al bag, mismo caso que bow simple
	def agregar(self, key, sentiment):
		if key in self.bag:
			if (sentiment == 0):
				self.featurevec_neg[ self.bag[key] ] += 1
			else:
				#print "Agregando key %s en posicion %d size %d" % (key, self.bag[key], len(self.featurevec_pos))
				self.featurevec_pos[ self.bag[key] ] += 1
		else:
			self.bag[key] = self.contador
			if (sentiment == 0):	
				self.featurevec_neg.append(1)
				self.featurevec_pos.append(0)
			else:
				self.featurevec_pos.append(1)
				self.featurevec_neg.append(0)
			self.contador += 1
			self.words.append(key)
			
			
	# Devuelve la frecuencia con la que aparece una key
	def frecuencia(self, key, sentiment):
		if key in self.bag:
			if (sentiment == 0):
				return self.featurevec_neg[ self.bag[key] ]
			else:
				return self.featurevec_pos[ self.bag[key] ]
		else:
			return 0
			
	# Devuelve todas las keys que se guardan en el bag
	def palabrasGuardadas(self):
		return self.bag.keys()
		
	# Devuelve el feature vector con las frecuencias
	def featureVector(self, sentiment):
		if (sentiment == 0):
			return self.featurevec_neg
		else:
			return self.featurevec_pos
	
	def wordVector(self):
		return self.words
	
	# Devuelve el contador, solo para probar. ELIMINAR ESTO
	def contadorEn(self):
		return self.contador
	
	# Devuelve true si la key esta en el bag
	def estaEnBag(self, key):
		if key in self.bag:
			return True
		else:
			return False
	
	# Devuelve la posicion del feature vector en la que se encuentra la key
	def posicionEnBag(self, key):
		if key in self.bag:
			return self.bag[key]
		else:
			return -1


bagPrueba = Bag_of_words()
print "Agrego la palabra good (pos) a la bag\n"
bagPrueba.agregar('good', 1)
print "La palabra good deberia ahora aparecer una vez. Su frecuencia es %d" % bagPrueba.frecuencia('good', 1)
print "La lista de palabras guardadas es %s \n" % bagPrueba.palabrasGuardadas()
print bagPrueba.featureVector(1)
print "Agrego mas palabras: bad (neg), awesome (pos), good (pos)\n"
bagPrueba.agregar('bad', 0)
bagPrueba.agregar('awesome', 1)
bagPrueba.agregar('good', 1)
print bagPrueba.featureVector(1)
print "La palabra good deberia tener frec 2. Su frecuencia es %d" % bagPrueba.frecuencia('good',1)
print "La lista de palabras guardadas es %s \n" % bagPrueba.palabrasGuardadas()
print "La palabra terrible (neg) deberia tener frec 0. Su frecuencia es %d" % bagPrueba.frecuencia('terrible',0)
print "Contador deberia estar en 3, esta en", bagPrueba.contadorEn()
print bagPrueba.featureVector(0)

bagPrueba.agregar('awesome', 1)
print "La palabra awesome deberia tener frec 2. Su frecuencia es %d" % bagPrueba.frecuencia('awesome', 1)
print "Feature vector deberia ser [good, bad, awesome], osea [2, 0, 2], y es:\n", bagPrueba.featureVector(1)
print bagPrueba.featureVector(0)


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


# FALTA SACAR PALABRAS CORTAS

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


#num_reviews = 15000

for i in xrange( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, num_reviews )   
    if ((train["sentiment"][i]) == 1):
		clean_train_reviews_pos.append( review_to_words( train["review"][i] ))
    if ((train["sentiment"][i]) == 0):
		clean_train_reviews_neg.append( review_to_words( train["review"][i] ))

bagNueva = Bag_of_words()

for lista in clean_train_reviews_pos:
	reviewsplit = lista.split()
	for word in reviewsplit:
		bagNueva.agregar(word, 1)
for lista in clean_train_reviews_neg:
	reviewsplit = lista.split()
	for word in reviewsplit:
		bagNueva.agregar(word, 0)

merge = zip(bagNueva.featureVector(1), bagNueva.wordVector())
for count, tag in ([(count, tag) for tag, count in merge])[1:30]:
    print(count, tag)

print "Total de palabras %d" % len(bagNueva.wordVector())


# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column

output = pd.DataFrame( data={ "palabra": bagNueva.wordVector(), "frecuencia pos": bagNueva.featureVector(1) , "frecuencia neg": bagNueva.featureVector(0)} )

# "frecuencia pos": (bagNueva.featureVector(1))[bagNueva.posicionEnBag(word)] , "frecuencia neg": (bagNueva.featureVector(0))[bagNueva.posicionEnBag(word)]
# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )


# Pruebo con los etiquetados

contador = 0
max_frec_pos = max(bagNueva.featureVector(1))
print "Max positivas: %d" % max_frec_pos
max_frec_neg = max(bagNueva.featureVector(0))
print "Max negativas: %d" % max_frec_neg

for j in range (num_reviews, 25000):
	if( (j+1)%1000 == 0 ):
		print "Review %d of 25000\n" % (j+1) 

	clean_reviews = review_to_words( train["review"][j] )
	clean_reviews = clean_reviews.split()
	# Si esta en la bag positiva +1
	# Si esta en la bag negativa -1
	
	# Probar con un umbral mas chico

	puntuacion = 0
	for word in clean_reviews:
		if bagNueva.estaEnBag(word):
			frec_pos = (bagNueva.featureVector(1))[bagNueva.posicionEnBag(word)]
			frec_neg = (bagNueva.featureVector(0))[bagNueva.posicionEnBag(word)]
			frec_total = frec_pos + frec_neg		
			
			# Tratar caso de -1 si no llega a estar en la bag
			# Tratar caso de frecuencias iguales
			
			setentaPorciento = frec_total*0.7
			cuarentaPorcientoPos = 0.001*max_frec_pos
			cuarentaPorcientoNeg = 0.001*max_frec_neg
			
			if ((frec_pos > setentaPorciento) and (frec_pos > cuarentaPorcientoPos)):
				puntuacion += 1
			elif ((setentaPorciento < frec_neg) and (frec_neg > cuarentaPorcientoNeg)):
				puntuacion -= 1
	if (((puntuacion > 0) and (	train["sentiment"][j]  == 1)) or ((puntuacion < 0) and (train["sentiment"][j] == 0)) ):
		contador += 1
print contador



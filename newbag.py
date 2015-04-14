#!/usr/bin/env python
# 
# Pruebas para un Bag Of Words. 
# 
# python newbag.py
# *************************************** #

import pandas as pd   
import numpy as np
from bs4 import BeautifulSoup  
import re
import nltk
from nltk.probability import ELEProbDist, FreqDist
from nltk import NaiveBayesClassifier

from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
import unicodedata
import pickle #Para persitencia

# # # # Implementando un Bag of Words decente

class Bag_of_words:
	# Un bow seria un diccionario que guarda word:elemarray
	# siendo elemarray la posicion en donde se guardara la frecuencia
	# en el feature vector generado en este proceso
	def __init__(self):
		self.bag = {}   
		self.featurevec_pos = []
		self.words = []
		self.featurevec_neg = [] 
		self.contador = 0 # Indica el indice para los feature vectors

	# Agrega una nueva key al bow, mismo caso que bag simple
	def agregar(self, key, sentiment):
		if key in self.bag:
			if (sentiment == 0):
				self.featurevec_neg[ self.bag[key] ] += 1
			else:
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
	
	# Devuelve el vector con las palabras sin repetir
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

def pruebaBag():
	print "\n# # # # Pruebo el Bag of Words decente \n"
	bagPrueba = Bag_of_words()
	print "Agrego la palabra good (pos) a la bag"
	bagPrueba.agregar('good', 1)
	print "La palabra good deberia ahora aparecer una vez. Su frecuencia es %d" % bagPrueba.frecuencia('good', 1)
	print "La lista de palabras guardadas es %s" % bagPrueba.palabrasGuardadas()
	print "El feature vector positivo es", bagPrueba.featureVector(1)
	print "\nAgrego mas palabras: bad (neg), awesome (pos), good (pos)"
	bagPrueba.agregar('bad', 0)
	bagPrueba.agregar('awesome', 1)
	bagPrueba.agregar('good', 1)
	print "El feature vector positivo es", bagPrueba.featureVector(1)
	print "La palabra good deberia tener frec 2. Su frecuencia es %d" % bagPrueba.frecuencia('good',1)
	print "La lista de palabras guardadas es %s" % bagPrueba.palabrasGuardadas()
	print "\nLa palabra terrible (neg) deberia tener frec 0. Su frecuencia es %d" % bagPrueba.frecuencia('terrible',0)
	print "Contador deberia estar en 3, esta en", bagPrueba.contadorEn()
	print "El feature vector negativo es", bagPrueba.featureVector(0)
	print "\nAgrego la palabra awesome (pos) a la bag"
	bagPrueba.agregar('awesome', 1)
	print "La palabra awesome deberia tener frec 2. Su frecuencia es %d" % bagPrueba.frecuencia('awesome', 1)
	print "Feature vector deberia ser [good, bad, awesome], osea [2, 0, 2], y es:\n", bagPrueba.featureVector(1)
	print "El feature vector negativo sigue siendo", bagPrueba.featureVector(0)
	print "\n"

# # # # Implementando el Algoritmo 1 
# Se usa un solo bag que contenga todas las palabras y dos feature vectors:
# uno positivo y otro negativo, ambos que contengan las frecuencias de
# aparicion de las palabras, almacenadas el la lista words.

# Paso 1. Limpio las reviews con sentiment. Se eliminan las stop words, mayusculas, etc

def review_to_words( raw_review ):
    # Remueve HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    # Remueve non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    # Lower case y split de palabras
    wordsSplit = letters_only.lower().split()
    words = wordsSplit # sacar esto para probar el Lemmatizer
    """# Uso el Lemmatizer
    words = []
    wordnet_lemmatizer = WordNetLemmatizer()
    for w in wordsSplit:
		lem = wordnet_lemmatizer.lemmatize(w)
		words.append(lem)
	# Uso Porter Stemming
    words = []
    st = LancasterStemmer()
    for w in wordsSplit:
		stem = st.stem(w)
		words.append(stem)
	"""
    # In Python, searching a set is much faster than searching
    # a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    # Remueve stop words
    meaningful_words = [w for w in words if not w in stops]   
    # Junta las palabras en un unico string
    return( " ".join( meaningful_words ))

# Paso 2. Se devuelve una bow entrenada 

clean_reviews = [] # una lista que contiene tuplas, las cuales contienen
				   # (lista de palabras de la review, sentiment de la review)
def entrenamientoBag():
	train = pd.read_csv("labeledTrainData.tsv", header=0, \
						delimiter="\t", quoting=3)
	# Si la review limpia es positiva, va a parar a la lista clean_train_reviews_pos
	# Caso contrario, va a la clean_train_reviews_neg
	clean_train_reviews_pos = []
	clean_train_reviews_neg = []
	
	#num_reviews = train["review"].size
	num_reviews = 20000
	
	print "# # # # Parseando reviews...\n"
	for i in xrange( 0, num_reviews ):
		if( (i+1)%2000 == 0 ):
			print "Review %d of %d\n" % ( i+1, num_reviews )   
		if ((train["sentiment"][i]) == 1):
			clean_train_reviews_pos.append( review_to_words( train["review"][i] ))
		if ((train["sentiment"][i]) == 0):
			clean_train_reviews_neg.append( review_to_words( train["review"][i] ))

	# Comienzo con el entrenamiento del bow
	bagNueva = Bag_of_words()

	for review in clean_train_reviews_pos:
		reviewsplit = review.split()
		nuevo = []
		for w in reviewsplit:
			w = unicodedata.normalize('NFKD', w).encode('ascii','ignore') 
			nuevo.append(w)
		reviewsplit = nuevo
		clean_reviews.append( (reviewsplit, 'positive') )
		
		for word in reviewsplit:
			bagNueva.agregar(word, 1)

	for review in clean_train_reviews_neg:
		reviewsplit = review.split()
		nuevoneg = []
		for w in reviewsplit:
			w = unicodedata.normalize('NFKD', w).encode('ascii','ignore') 
			nuevoneg.append(w)
		reviewsplit = nuevoneg
		clean_reviews.append( (reviewsplit, 'negative') )

		for word in reviewsplit:
			bagNueva.agregar(word, 0)
	# Termina el entrenamiento del bow
	
	# Paso los resultados a un nuevo csv
	output = pd.DataFrame( data={ "palabra": bagNueva.wordVector(), \
			"frecuencia pos": bagNueva.featureVector(1) ,           \
			"frecuencia neg": bagNueva.featureVector(0)} )

	output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )
	return bagNueva

# Paso 3. Pruebo resultados con la idea del +1 y -1

def prueboMasMenosUno(bagNueva):
	print "\n\n# # # # Pruebo mas y menos uno con la bag ya entrenada \n"
	# Pruebo con las reviews con sentiment, solo para ver si funciona
	train = pd.read_csv("labeledTrainData.tsv", header=0, \
						delimiter="\t", quoting=3)	
	# Entrene con las primeras 20000, pruebo con las otras 5000
	num_reviews = 20000
	
	max_frec_pos = max(bagNueva.featureVector(1))
	max_frec_neg = max(bagNueva.featureVector(0))

	contador = 0	# Cuenta la cantidad de casos acertados
	for j in range (num_reviews, 25000):
		if( (j+1)%1000 == 0 ):
			print "Review %d of 25000\n" % (j+1) 
		clean_reviews = review_to_words( train["review"][j] )
		clean_reviews = clean_reviews.split()
		
		puntuacion = 0	# Esta es la puntuacion de la review i
		for word in clean_reviews:
			if bagNueva.estaEnBag(word):
				frec_pos = (bagNueva.featureVector(1))[bagNueva.posicionEnBag(word)]
				frec_neg = (bagNueva.featureVector(0))[bagNueva.posicionEnBag(word)]
				frec_total = frec_pos + frec_neg		
				# Probar con un umbral mas chico		
				# Tratar caso de -1 si no llega a estar en la bag
				# Tratar caso de frecuencias iguales	
				setentaPorciento = frec_total*0.7
				cuarentaPorcientoPos = 0.001*max_frec_pos
				cuarentaPorcientoNeg = 0.001*max_frec_neg
				
				if ((frec_pos > setentaPorciento)): #and (frec_pos > cuarentaPorcientoPos)):
					puntuacion += 1
				elif ((frec_neg > setentaPorciento)): #and (frec_neg > cuarentaPorcientoNeg)):
					puntuacion -= 1
		
		if (((puntuacion > 0) and (	train["sentiment"][j]  == 1)) or ((puntuacion < 0) and (train["sentiment"][j] == 0)) ):
			contador += 1
			
	print "%d of %d casos acertados" %(contador, 25000 - num_reviews)


# # # # Implementando el Algoritmo 2
# Pruebo con un Naive Bayes para ver el porcentaje de aciertos
word_features = []

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def getKey(item):
	return item[0]

def naiveBayes(bagNueva):
	# Obtengo una lista con tuplas (frecuencia, palabra)
	merge = zip(bagNueva.featureVector(1), bagNueva.wordVector())
	# Ordeno por frecuencia de mayor a menor
	mergeOrdenado = sorted(merge, key=getKey, reverse=True)
	for tupla in mergeOrdenado:
		word_features.append(tupla[1])

	training_set = nltk.classify.apply_features(extract_features, clean_reviews)

	classifier = nltk.NaiveBayesClassifier.train(training_set)

	# PROBANDO
	reviewsPruebas = []
	sentimentPruebas = []
	train = pd.read_csv("labeledTrainData.tsv", header=0, \
					delimiter="\t", quoting=3)

	inicio = 10001
	fin = 25000

	for j in range (inicio, fin):
		reviewsPruebas.append( review_to_words( train["review"][j] ) )
		sentimentPruebas.append( train["sentiment"][j] )

	i = 0
	porcentaje = 0
	for rev in reviewsPruebas:
		sentimiento = classifier.classify(extract_features( rev.split() ) )
		if ((sentimiento == 'negative' ) and (sentimentPruebas[i] == 0)):
			porcentaje += 1 
		if ((sentimiento == 'positive' ) and (sentimentPruebas[i] == 1)): #No corresponde un elif?
			porcentaje += 1
		i += 1

	total = porcentaje / ((fin - inicio) + 0.0)
	print porcentaje 
	print total

def clasificarDesdeArchivo():
	reviewsPruebas = []
	sentimentPruebas = []
	train = pd.read_csv("labeledTrainData.tsv", header=0, \
					delimiter="\t", quoting=3)
	classifier = levantarDeArchivo()


# # # # Implementando el Algoritmo 3
# Pruebo con un Max Entropy Classifier para ver el porcentaje de aciertos

# def maxEntropy(bagNueva): 
# 	# Obtengo una lista con tuplas (frecuencia, palabra)
# 	merge = zip(bagNueva.featureVector(1), bagNueva.wordVector())
# 	# Ordeno por frecuencia de mayor a menor
# 	mergeOrdenado = sorted(merge, key=getKey, reverse=True)
# 	for tupla in mergeOrdenado:
# 		word_features.append(tupla[1])
# 	# training_set es una lista de tuplas. Cada tupla contiene un feature
# 	# dictionary y el sentiment para cada review 
# 	training_set = nltk.classify.apply_features(extract_features, clean_reviews)

# 	MaxEntClassifier = nltk.classify.maxent.MaxentClassifier.train(training_set, 'GIS', trace=3, \
# 						encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 10)
# 	testTweet = 'Larry is my friend'
# 	print testTweet
# 	processedTestTweet = review_to_words(testTweet)	
# #print MaxEntClassifier.classify(extract_features(processedTestTweet.split()))

def persistir(classifier):
	f = open('naiveBayes.pickle', 'wb')
	pickle.dump(classifier, f)
	f.close()

def levantarDeArchivo():
	f = open('naiveBayes.pickle')
	classifier = pickle.load(f)
	f.close()
	return classifier

def main():

	reviewsPruebas = [] 
	sentimentPruebas = []
	train = pd.read_csv("labeledTrainData.tsv", header=0, \
					delimiter="\t", quoting=3)
	#pruebaBag()
	bag = entrenamientoBag()
	print "Total de palabras en la bag: %d" % len(bag.wordVector())
	print "Frecuencia maxima de palabra de las reviews positivas: %d" % max(bag.featureVector(1))
	print "Frecuencia maxima de palabra de las reviews negativas: %d" % max(bag.featureVector(0))
	prueboMasMenosUno(bag)
	#tuvieja = naiveBayes(bag)
	#persistir(tuvieja)
	#maxEntropy(bag)
	#clasificarDesdeArchivo()
	


if __name__ == "__main__":
    main()

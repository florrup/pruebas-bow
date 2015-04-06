#!/usr/bin/env python
# 
# Twitter sentiment analysis. NAIVE BAYES.
# 
# python twitter.py
# *************************************** #

import nltk
from nltk.probability import ELEProbDist, FreqDist
from nltk import NaiveBayesClassifier

# Listado de tweets con sentiment positivo
pos_tweets = [('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive')]
              
# Listado de tweets con sentiment negativo
neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative')]
              
# Listado de todos los tweets. Por cada tweet hay una tupla:
# (listado de palabras, sentiment)
tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
    tweets.append((words_filtered, sentiment))

# Listado con los tweets a probar, mismo formato que la lista 'tweets'
test_tweets = [
    (['feel', 'happy', 'this', 'morning'], 'positive'),
    (['larry', 'friend'], 'positive'),
    (['not', 'like', 'that', 'man'], 'negative'),
    (['house', 'not', 'great'], 'negative'),
    (['your', 'song', 'annoying'], 'negative')]

# Devuelve un listado con todas las palabras en la lista 'tweets'
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

# Devuelve un listado con cada palabra ordenada por frecuencia (mayor a menor)
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(get_words_in_tweets(tweets))

# Devuelve un diccionario que indica que palabras se encuentran en el document
# pasado por parametro. En este caso, el parametro es el tweet. 
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

# Contiene los feature sets etiquetados. Es una lista de tuplas de la forma:
# (feature dictionary, sentiment) de cada tweet.
training_set = nltk.classify.apply_features(extract_features, tweets)

# Entreno al classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

# Pruebo el classifier con un tweet
tweet = 'Larry is my friend'
print classifier.classify(extract_features(tweet.split()))

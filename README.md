# pruebas-bow

python newbag.py

Bag_of_Words_model.csv es la bag final, entrenada con todas las 25.000 reviews etiquetadas de labeledTrainData.tsv

- - - - 
Entrenando con 20.000 reviews, probando con 5.000

- En el caso de mas y menos unos (sumo uno si la palabra es positiva, resto uno si es negativa), hay 4.151 / 5.000 aciertos
- Usar un Lemmatizer baja la cantidad de aciertos: 4.043 / 5.000

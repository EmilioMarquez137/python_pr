# -*- coding: utf-8 -*-
"""word_sense_des.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oypUD_9m93kYq7K3I6AWHIlJ1fYO0X4o
"""

from nltk import wsd
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from spacy.cli import download
from spacy import load
import warnings

nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('wordnet2022')
nlp = load('en_core_web_sm')

! cp -rf /usr/share/nltk_data/corpora/wordnet2022 /usr/share/nltk_data/corpora/wordnet # temp fix for lookup error.

"""En el siguiente ejemplo la palabra morir tiene un significado diferente en cada frase.
Sólo entendiendo el contexto de la palabra la PNL puede improvisar más.
"""

X = 'The die is cast.'
Y = 'Roll the die to get a 6.'
Z = 'What is dead may never die.'

""" En este ejemplo se  va utilizar wordnet de la universidad de princeton para obtener de la palabra diferentes frases de contexto como parte de la oración adjunta.
 Wordnet : es una base de datos léxica del inglés. Sustantivos, verbos, adjetivos y adverbios que se agrupan en conjuntos de  sinónimos cognitivos (synsets), cada uno de los cuales expresa un concepto distinto.
 En python los datos de wordnet se cargan con NLTK.
 Por ejemplo se pasa la palabra die a wordnet y se intenta obtener las diferentes frases unqiue wordnet que se tiene para die
 La salida de wordnet puede ser definiciones diferentes de die que incluyen verbos y sustantivos

"""

wn.synsets("die")

# comprobar detalles relacionados con el sustantivo
wn.synsets('die', pos=wn.NOUN)

# imprimir todas las definiciones relacionados con  sustantivos
i = 0
for syn in wn.synsets('die', pos=wn.NOUN):
    print("definicion {0} : {1}".format(i, syn.definition()))
    i += 1

# imprimir todas las definiciones relacionados con verbos
i =0
for syn in wn.synsets('die', pos=wn.VERB):
    print("defination {0} : {1}".format(i, syn.definition()))
    i+=1

"""Word-Sense Disambiguation con Algoritmo Lesk"""

# introduzca la frase X "La suerte está echada" y compruebe si lesk es capaz de encontrar la frase similar correcta.
print(X)
print(wsd.lesk(X.split(), 'die'))
print(wsd.lesk(X.split(), 'die').definition())

"""Para la frase de entrada X,lesk ha encontrado una frase similar coincidente cuyo tipo es verbo, y que no es correcta. En la siguiente se pasará explícitamente la parte de la oración (POS) y se comprobará la salida

La POS tagging (Part of Speech" en inglés) es un proceso en el que cada palabra en un texto es etiquetada con su categoría gramatical, como sustantivo, verbo, adjetivo, etc.
"""

# al pasar un POS conseguimos la definición correcta
print(X)
wsd.lesk(X.split(), 'die', pos=wn.NOUN).definition()

# para la frase X, es decir, "Tira el dado para obtener un 6." es de nuevo un sustantivo.
print(Y)
wsd.lesk(Y.split(), 'die').definition()

# Con pasar un POS conseguimos la defianción correcta?
wsd.lesk(Y.split(), 'die', pos=wn.NOUN).definition()

# Observaciones similares con la sentencia Z
print(Z)
wsd.lesk(Z.split(), 'die').definition()

wsd.lesk(Z.split(), 'die', pos=wn.VERB).definition()

"""Etiquetado automático"""

# Pasar una frase a spacy para la búsqueda automática de la etiqueta POS.
# Observar la salida, es evidente que la forma POS se encuentra para una frase.

sent1 = "I loved the screen on this phone."
doc1 = nlp(sent1)
for tok in doc1:
    print(tok.text,tok.pos_)

# ejemplos adicionales
sent2 = "The battery life on this phone is great. !"
doc2 = nlp(sent2)
for tok in doc2:
    print(tok.text,tok.pos_)

# leer el etiquetado POS y el lema (sólo ampliando un ejemplo para mostrar el lema)

sent1 = "I loved the screen on this phone."
doc1 = nlp(sent1)
pos = []
lemma = []
text = []
for tok in doc1:
    pos.append(tok.pos_)
    lemma.append(tok.lemma_)
    text.append(tok.text)
nlp_table = pd.DataFrame({'text':text,'lemma':lemma,'pos':pos})
nlp_table.head()

"""Etiquetado Automatico POS + Lesk con spaCy"""

POS_MAP = {
    'VERB': wn.VERB,
    'NOUN': wn.NOUN,
    'PROPN': wn.NOUN
}


def lesk(doc, word):
    found = False
    for token in doc:
        if token.text == word:
            word = token
            found = True
            break
    if not found:
        raise ValueError(f'Word \"{word}\" No aparece en el documentos: {doc.text}.')
    pos = POS_MAP.get(word.pos_, False)
    if not pos:
        warnings.warn(f'POS tag para {word.text} no encontrada en wordnet. Volver al comportamiento por defecto de Lesk.')
    args = [c.text for c in doc], word.text
    kwargs = dict(pos=pos)
    return wsd.lesk(*args, **kwargs)

doc = nlp('Roll the die to get a 6.')
lesk(doc, 'die')

# búsqueda de la etiqueta POS por defecto, para ayudar a lesk a encontrar la definición correcta.

lesk(doc, 'die').definition()

# ejemplo adicional, con la sentencia siguiente

lesk(nlp('I work at google.'), 'google').definition()

# revisar cuando se trata como verbo google

lesk(nlp('I will google it.'), 'google').definition()

# esperanza como sustantivo

lesk(nlp('Her pep talk gave me hope'), 'hope').definition()

# esperanza como verbo

lesk(nlp('I hope we win!'), 'hope').definition()
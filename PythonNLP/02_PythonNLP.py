import nltk

text = "Mary had a little lamb. Her fleece was white as snow"


# tokenization - splitting text into smaller chunks
# e.g.  sentences, words
print('Tokenization')
from nltk.tokenize import word_tokenize, sent_tokenize
sentences = sent_tokenize(text)
print(sentences)
words = [word_tokenize(sentence) for sentence in sentences]
print(words)


# StopWords - very common words that don't add to the meaning of the text
# ex: a, an, the, as
# also punctuation
print('StopWords')
from nltk.corpus import stopwords
from string import punctuation
customStopWords = set(stopwords.words('english') + list(punctuation))
wordsWOStopwords = [word for word in word_tokenize(text) if word not in customStopWords]
print(wordsWOStopwords)


#
print('Bigrams')
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(wordsWOStopwords)
bigrams = sorted(finder.ngram_fd.items())
print(bigrams)


# Stemming - reducing a word to its base
# e.g.  - closed, closing, close => clos
# This helps cut down on 'false duplicates'
print('Stemming')
from nltk.stem.lancaster import LancasterStemmer
text2 = "Mary closed on closing night when she was in the mood to close."
st = LancasterStemmer()
stemmedWords = [st.stem(word) for word in word_tokenize(text2)]
print(stemmedWords)


# Tagging - Identifying which Part of Speech (POS) a word is, given the context
print('Tagging')
tags = nltk.pos_tag(word_tokenize(text2))
print(tags)


# Lexicon - identifies which version of a word is being used
# e.g. Bass
# - lowest part of the musical range
# - adult male singer with the lowest voice
# - Musical instrument
# - Fish
# - 
print('Lexicon')
from nltk.wsd import lesk
sense1 = lesk(word_tokenize("Sing in a lower tone, along with the bass"), 'bass')
print(sense1, sense1.definition())

sense2 = lesk(word_tokenize("This sea bass was really hard to catch"), 'bass')
print(sense2, sense2.definition())

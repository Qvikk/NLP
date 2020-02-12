import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import seaborn as sns
from gensim.models import Word2Vec


data = pd.read_csv('C:/Users/arist/PycharmProjects/lab1/bbc-text.csv')
print(data.head())
data['category_id'] = data['category'].factorize()[0]

colslist = ['type', 'news', 'category_id']
data.columns = colslist

text_file = open("stopwords.txt", "r")
stopwords = text_file.read().split('\n')


data['news_without_stopwords'] = data['news'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
print(len(data['news_without_stopwords'][0]))
print(data['news_without_stopwords'])

ps = PorterStemmer()

data['news_porter_stemmed'] = data['news_without_stopwords'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
print(data['news_without_stopwords'][0])
print(data['news_porter_stemmed'][0])
data['news_porter_stemmed'] = data['news_porter_stemmed'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
print(data['news_porter_stemmed'][0])
data['news_porter_stemmed'] = data['news_porter_stemmed'].str.replace('[^\w\s]','')
freq = pd.Series(' '.join(data['news_porter_stemmed']).split()).value_counts()
print(freq.head())

freq2 = freq[freq <= 3]
print(freq2)

freq3 = list(freq2.index.values)
print(freq3)

w2v_model = Word2Vec(min_count=200,window=5,size=100,workers=4)
sentences = str(data['news_porter_stemmed'])

print(type(sentences))

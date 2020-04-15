# encoding: utf-8
# -*- coding: cp1252 -*-.
#!/usr/bin/python

import sys
import os
import re
import nltk
import string
import PyPDF2
import textract
import random
import rouge
import pandas as pd
import numpy as np
import seaborn as sns
import preprocessor as p
import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt
from rouge import Rouge
from fuzzywuzzy import fuzz
from time import time, sleep
from textblob import TextBlob
from nltk.corpus import stopwords 
from sklearn.svm import OneClassSVM
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score,recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans,DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from scipy.stats import ttest_1samp,ttest_ind, ttest_rel, norm, pearsonr
 
def pdf_txt_converter(filename,outputFile='article.txt'):

	filename='artigo.pdf'
	outputFile='article.txt'	
	
	output = "output-"+"".join([chr(int(65+random.random()*25)) for i in range(0,10)])

	os.popen("pdftoppm "+filename+" " + output +" -png").read()
	pages = int(os.popen("ls *"+output+"* | wc -l").read()) 

	texto = ""

	for i in range (1,pages+1):
		if i < 10: p = "0"+str(i)
		else: p = str(i)
		os.popen("tesseract "+ output+"-"+p+".png texto-"+output+"-"+p+" -l por").read()
		texto += open("texto-"+output+"-"+p+".txt").read()

	os.popen("rm -rf *"+output+"*").read()
	open(outputFile,"w").write(texto)

	'''
	resumo = texto.find ("Resumo.")
	abstract = texto.find("Abstract.")
	intro = texto.find("1. Introdu")

	if abstract != -1 and resumo > abstract:
		texto_resumo = texto[resumo:intro]
	else:
		texto_resumo = texto[resumo:abstract]
	 
	print (texto_resumo)
	'''

def text_reading():
	article = open('article.txt','r')
	file = article.readlines()
	article.close()
	contiguous_string = ''
	for text in file:
		contiguous_string += text.strip('\n')
	abstract, body = contiguous_string.split('Introdução')
	'''
	print('Abstract: \n')
	print (abstract)
	print('Body: \n')
	print (body)
	'''
	sentences = body.split('.')
	COLS = ['Sentences']
	database=pd.DataFrame(sentences,columns=COLS)
	csvFile = open("database.csv", 'w' ,encoding='utf-8')
	database.to_csv(csvFile, mode='w', columns=COLS, index=False, encoding="utf-8")

def remove_punctuation(word):

	for ch in string.punctuation:
		word = word.replace(ch, "")
	if(len(word) < 2 or word.isdigit()):
		return ""
	return word

def similarity(word, names):

	try:
		for w in names[word[0]]:
			if(fuzz.ratio(w, word) > 80):
				return True
	except KeyError:
		pass
	return False

def get_name_dict():

	names = open("names.txt", encoding="utf-8")
	names_lines = names.readlines()
	names_dict = {}
	for lines in names_lines:
		line = lines.replace('\n','').lower()
		if line[0] not in names_dict: names_dict[line[0]] = []
		names_dict[line[0]].append(line)
	return names_dict

def position_weighted_metric():
	
	df = pd.read_csv("cleaned_database.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	count_vectorizer = CountVectorizer(ngram_range=(1,1))
	tfidf_matrix=tfidf_vectorizer.fit_transform(df["Cleaned_Sentences"].values.astype('U')).todense()
	count_matrix=count_vectorizer.fit_transform(df["Cleaned_Sentences"].values.astype('U')).todense()
	'''Assume general form: (A,B) C
	A: Document index 
	B: Specific word-vector index 
	C: TFIDF score for word B in document A'''
	position_measures=[]
	for row in range(len(tfidf_matrix)):
		position_measures.append(1-((row)/len(tfidf_matrix)))
	df['Position'] = position_measures

	COLS = ['Sentences', 'Cleaned_Sentences','Position']
	csvFile = open("cleaned_database.csv", 'w' ,encoding='utf-8')
	df.to_csv(csvFile, mode='w', columns=COLS, index=False, encoding="utf-8")

def tfidf_metric():

	df = pd.read_csv("cleaned_database.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	count_vectorizer = CountVectorizer(ngram_range=(1,1))
	tfidf_matrix=tfidf_vectorizer.fit_transform(df["Cleaned_Sentences"].values.astype('U')).todense()
	count_matrix=count_vectorizer.fit_transform(df["Cleaned_Sentences"].values.astype('U')).todense()

	tfidf_measures=[]
	for row in range(len(tfidf_matrix)):
		tfidf_measures.append(sum(np.asarray(tfidf_matrix)[row])/len(np.asarray(tfidf_matrix)[row]))
	df['TDIDF_Average'] = tfidf_measures

	COLS = ['Sentences', 'Cleaned_Sentences','Position','TDIDF_Average']
	csvFile = open("cleaned_database.csv", 'w' ,encoding='utf-8')
	df.to_csv(csvFile, mode='w', columns=COLS, index=False, encoding="utf-8")

def tfidf_cossine_euclidean():
	
	df = pd.read_csv("cleaned_database.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	count_vectorizer = CountVectorizer(ngram_range=(1,1))
	tfidf_matrix=tfidf_vectorizer.fit_transform(df["Cleaned_Sentences"].values.astype('U')).todense()
	count_matrix=count_vectorizer.fit_transform(df["Cleaned_Sentences"].values.astype('U')).todense()

	euclidean_measures=[]
	cosine_measures=[]
	for index in range(len(tfidf_matrix)):
		euclidean=0
		cosine=0
		for row in tfidf_matrix:
			euclidean+=euclidean_distances(tfidf_matrix[index],row)
			cosine+=cosine_similarity(tfidf_matrix[index],row)

		euclidean=float(str(euclidean).replace(']]', '').replace('[[', '')) #Convert a single value matrix into a float 
		cosine=float(str(cosine).replace(']]', '').replace('[[', '')) #Convert a single value matrix into a float
		euclidean_measures.append(euclidean/len(tfidf_matrix))
		cosine_measures.append(cosine/len(tfidf_matrix))

	df['Euclidean Average'] = euclidean_measures
	df['Cosine Average'] = cosine_measures
	#print (df['Cosine Average'])
	#print (df['Euclidean Average'])

	COLS = ['Sentences', 'Cleaned_Sentences','Position','TDIDF_Average','Euclidean Average','Cosine Average']
	csvFile = open("cleaned_database.csv", 'w' ,encoding='utf-8')
	df.to_csv(csvFile, mode='w', columns=COLS, index=False, encoding="utf-8")

def brush_path():

	df = pd.read_csv("cleaned_database.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	count_vectorizer = CountVectorizer(ngram_range=(1,1))
	tfidf_matrix=tfidf_vectorizer.fit_transform(df["Cleaned_Sentences"].values.astype('U')).todense()
	count_matrix=count_vectorizer.fit_transform(df["Cleaned_Sentences"].values.astype('U')).todense()

	G = nx.Graph()
	G.add_nodes_from(df.index)
	print 
	pass
	for index in range(len(tfidf_matrix)):
		cosine=0
		for row in range(len(tfidf_matrix)):
			cosine=cosine_similarity(tfidf_matrix[index],tfidf_matrix[row])
			cosine=float(str(cosine).replace(']]', '').replace('[[', ''))
			#if index == 30:
				#print(cosine)

			if (cosine>=0.1) and (cosine!=1.0) :
				G.add_weighted_edges_from([(index,row,cosine)])

	#pos = nx.spring_layout(G)
	pos = nx.fruchterman_reingold_layout(G)
	labels = nx.get_edge_attributes(G,'weight')
	#pos = nx.circular_layout(G)
	#pos = nx.shell_layout(G)

	nx.draw(G,pos, with_labels=True, font_weight='bold')
	#plt.show()

	degree=[]
	betweenness=[]
	for value in nx.betweenness_centrality(G).values():
		betweenness.append(value)
	for value in nx.degree_centrality(G).values():
		degree.append(value)

	df['Betweenness'] = betweenness
	df['Degree'] = degree

	#print(nx.number_connected_components(G)) #3
	components=[]
	for component in nx.connected_components(G):
		components.append(component)

	names=[]
	for c in range(len(components)):
		values=[]
		for index in range(len(betweenness)):
			if index in components[c]:
				values.append(betweenness[index])
			else:
				values.append(0)
		names.append('Feature '+str(c))
		df[names[c]] = values
		#print(df[names[c]])

	COLS = ['Sentences', 'Cleaned_Sentences','Position','TDIDF_Average','Euclidean Average','Cosine Average', 'Betweenness','Degree']
	for i in range(len(names)):
		COLS.append(names[i])

	csvFile = open("cleaned_database.csv", 'w' ,encoding='utf-8')
	df.to_csv(csvFile, mode='w', columns=COLS, index=False, encoding="utf-8")

def k_medoids_method():

	#X = np.asarray([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])
	database = pd.read_csv("cleaned_database.csv",header=0)
	df = database.drop(['Sentences', 'Cleaned_Sentences'], axis=1)
	kmedoids = KMedoids(n_clusters=5, random_state=0).fit(df)
	#print(kmedoids.labels_)
	#pred=kmedoids.predict([[0,0], [4,4]])
	#print(kmedoids.cluster_centers_)
	centers = pd.DataFrame(kmedoids.cluster_centers_, columns=df.columns)
	#print (centers)
	
	#index=df.loc[kmedoids.cluster_centers_].index()
	#print(kmedoids.inertia_)

	output=database.loc[df['Position'].isin(centers['Position'])]
	output_abstract='.'.join(output['Sentences'].to_list())
	#print (output_abstract)

	return output_abstract
	
def rouge_evaluation(output_abstract):

	article = open('article.txt','r')
	file = article.readlines()
	article.close()
	contiguous_string = ''
	for text in file:
		contiguous_string += text.strip('\n')
	reference_abstract, body = contiguous_string.split('Introdução')

	rouge = Rouge()
	scores = rouge.get_scores(output_abstract, reference_abstract)
	print (scores)

	print('Resumo')
	print(output_abstract)

def preprocessing(names_dict):

	stemmer = nltk.stem.RSLPStemmer()
	stop_words = set(stopwords.words('portuguese'))
	
	df = pd.read_csv("database.csv",header=0)

	df["Cleaned_Sentences"] = df["Sentences"].str.lower()
	df["Cleaned_Sentences"] = df["Cleaned_Sentences"].apply(lambda x:' '.join(remove_punctuation(word) for word in word_tokenize(str(x))))
	df["Cleaned_Sentences"] = df["Cleaned_Sentences"].apply(lambda x:' '.join(word for word in word_tokenize(x) if word not in stop_words))
	df["Cleaned_Sentences"] = df["Cleaned_Sentences"].apply(lambda x:' '.join(stemmer.stem(word) for word in word_tokenize(x) if not similarity(word, names_dict)))

	COLS = ['Sentences', 'Cleaned_Sentences']
	csvFile = open("cleaned_database.csv", 'w' ,encoding='utf-8')
	df.to_csv(csvFile, mode='w', columns=COLS, index=False, encoding="utf-8")
	
if __name__ == "__main__":

	filename = 'artigo.pdf'
	pdf_txt_converter(filename)
	text_reading()
	names_dict = get_name_dict()
	preprocessing(names_dict)
	position_weighted_metric()
	tfidf_metric()
	tfidf_cossine_euclidean()
	#text_rank_method()
	brush_path()
	output = k_medoids_method()
	rouge_evaluation(output)

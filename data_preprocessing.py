#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Kerian Thuillier"
__email__ = "kerian.thuillier@ens-rennes.fr"

####################################################################################
#
import os
import re
import pandas as pd
from sklearn import metrics # metrics.confusion_matrix(y_pred, y_pred_class_tfidf)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn import svm


####################################################################################
#	FILES TO LIST OF STRING
#

def file_to_string (filename) :
	with open(filename, 'r') as myfile:
		data = myfile.read()
		# Filter the markdown comments
		data = data.replace(r'[\W]', "")
		data = data.replace('_', "")
		data = re.sub(r'(\d)\s+(\d)', r'\1\2', data)
	return data

def folder_to_list (path) :
	result = []
	for file in os.listdir(path) :
		if file.endswith(".txt") :
			result.append(file_to_string(path + "/" + file))
	return result

def zip_with_class_label (l, class_label) :
	data_x = l
	data_y = (class_label for i in range(0, len(l)))
	data = zip(data_x, data_y)
	return list(data)

def create_datafram (neg_path, pos_path) :
	data_neg = folder_to_list(neg_path)
	data_neg = zip_with_class_label(data_neg, "negative")

	data_pos = folder_to_list(pos_path)
	data_pos = zip_with_class_label(data_pos, "positive")

	data = data_neg + data_pos
	dataframe = pd.DataFrame(data, columns = ['Review' , 'Label'])

	return dataframe

####################################################################################


####################################################################################
#	DATA VECTORIZATION
#

def vectorize_data (data) :
	vector = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.5, min_df=2)
	tf_transformer = TfidfTransformer(use_idf=True)

	vector.fit(data)
	vectorized_data = vector.transform(data)
	vectorized_data = tf_transformer.fit_transform(vectorized_data)

	return vectorized_data

####################################################################################

####################################################################################
#	COMPUTE DATA SET
#

def compute_dataset (df, train_proportion, validate_proportion) :
	X = df.Review
	Y = df.Label.map({"negative":0, "positive":1})
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_proportion, shuffle=True, random_state=1)
	X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size=validate_proportion, shuffle=True, random_state=1)
	return X_train, X_validate, X_test, Y_train, Y_validate, Y_test

####################################################################################

####################################################################################
#	MODEL -> SVM
# 


####################################################################################

df = create_datafram("./neg", "./pos")
X_train, X_validate, X_test, Y_train, Y_validate, Y_test = compute_dataset(df, 0.8, 0.2)

assert(X_train.shape == Y_train.shape)
assert(X_validate.shape == Y_validate.shape)
assert(X_test.shape == Y_test.shape)

X_train_vect = vectorize_data(X_train)
X_validate_vect = vectorize_data(X_validate)
X_test_vect = vectorize_data(X_test)

####################################################################################
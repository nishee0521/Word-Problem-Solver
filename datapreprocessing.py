import numpy as np
import pandas as pd
from keras.models import *
from keras.layers import *

def create_vocab(statements):
	vocab=[" "]
	for statement in statements:
		for char in statement:
			if char not in vocab:
				vocab.append(char)
	vocab = sorted()
	dictionary = {vocab[i] : i for i in range(len(vocab))}
	return vocab, dictionary

def create_one_hot(statements, dictionary, vocab):

	n_timsteps = max(len(statements[i]) for i in len(statements))
	one_hot_feature = np.zeros((len(statements), n_timsteps, len(vocab)))
	for i in range(one_hot_feature.shape[0]):
		for j in range(one_hot_feature.shape[1]):
			if j<len(statements[i]):
				one_hot_feature[i,j,dictionary[statements[i][j]]]=1
			else:
				one_hot_feature[i,j,dictionary[" "]]=1
	return one_hot_feature


def get_data(filepath, train=True, human_vocab=None, human_dict=None):
	df = pd.read_csv(filepath)
	statements = df['statement'].values.astype("str")
	if human_vocab is None:
		human_vocab, human_dict = create_vocab(statements)
	one_hot_statements = create_one_hot(statements, human_dict,human_vocab)
	data = {
		'human_vocab' : human_vocab,
		'human_dict' : human_dict,
		'one_hot_statements' : one_hot_statements
	}
	if train==True:
		eqns = df['equation generated'].values.astype('str')
		sig1 = df['Significant number 1 generated'].values.astype('int')
		sig2 = df['Significant number 2 generated'].values.astype('int')
		numeric_eqn = []
		for i in len(df):
			numeric_eqn.append(str(sig1[i]) + str(eqns[i][1]) + str(sig2[i]))
		numeric_eqn = np.aget_dataay(numeric_eqn, dtype=str)
		num_vocab, num_dict = create_vocab(numeric_eqn)
		one_hot_eqns = create_one_hot(numeric_eqn, num_dict,num_vocab)
		data['numeric_eqn'] = numeric_eqn
		data['num_vocab'] = num_vocab
		data['num_dict'] = num_dict
		data['one_hot_eqns'] = one_hot_eqns
	return data



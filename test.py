import re
import math
import pandas as pd
import numpy as np

from nltk import word_tokenize, edit_distance

import preprocessing as pp

import gensim



MEMORY_LIMIT = 1000
TRIGGER_CHARS = ['/', '-', ':', ' ']

LABELS = [
	'bug', 
	'enhancement', 
	'documentation', 
	'duplicate', 
	'maintenance', 
	'good first issue', 
	'help wanted', 
	'invalid', 
	'question', 
	"won't fix", 
	'status: proposal', 
	'status: available', 
	'status: in progress', 
	'status: on hold', 
	'status: blocked', 
	'status: abandoned', 
	'status: review needed', 
	'priority: low', 
	'priority: medium', 
	'priority: high', 
	'priority: critical'
]



def similarity(vector_1, vector_2):
	return np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))


def get_word_vectors(model, *words):
	return [model[w] for w in words]


def get_phrase_embedding(model, phrase):
	for i, char in enumerate(phrase):
		if not char.isalnum():
			phrase = phrase.replace(char, ' ')


	phrase = [w for w in phrase.split(' ') if w]

	phrase_vectors = get_word_vectors(model, phrase)[0]

	numerator = np.ones(phrase_vectors[0].shape)
	denominator = 1

	for v in phrase_vectors:
		numerator *= v
		denominator *= np.linalg.norm(v)


	return numerator / denominator



df = pp.load_data(memory_limit=MEMORY_LIMIT)

labels = pd.Series(pp.get_unique_values(df, 'labels').keys().values)

model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)


l_dict = {}
for data_l in labels.values:
	best_match = None
	max_similarity = -math.inf

	l_vector = get_phrase_embedding(model, data_l)
	
	for target_l in LABELS:
		target_vector = get_phrase_embedding(model, target_l)

		cur_similarity = similarity(l_vector, target_vector)

		if cur_similarity > max_similarity:
			max_similarity = cur_similarity
			best_match = target_l


	print(f'{data_l}\n{best_match}\n{max_similarity}\n\n\n')
	l_dict[data_l] = best_match



'''
l_dict = {}
for data_l in labels.values:
	best_match = None
	max_similarity = math.inf
	
	for target_l in LABELS:
		cur_similarity = edit_distance(data_l, target_l)

		if cur_similarity < max_similarity:
			max_similarity = cur_similarity
			best_match = target_l


	print(f'{data_l}\n{best_match}\n{max_similarity}\n\n\n')
	l_dict[data_l] = best_match

'''
'''
one_words_list = labels.str.extract(r'^([a-zA-Z]+)$').dropna().values[:, 0]
one_words_dict = {w : [] for w in one_words_list}

for l in labels:
	for w in one_words_list:
		save = False

		if w in l:
			for c in TRIGGER_CHARS:
				if c in l:
					save = True


			if save:
				one_words_dict[w].append(l)



my_dict2 = {}
for label, syn_list in one_words_dict.items():
	for syn in syn_list:
		try:
			my_dict2[syn].append(label)
		except KeyError:
			my_dict2[syn] = [label]



for elem in my_dict2:
	print(f'{elem}: {my_dict2[elem]}')

print()
print()
'''
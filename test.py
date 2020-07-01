import math
import pandas as pd
import numpy as np

import preprocessing as pp

from gensim.models import Word2Vec

import multiprocessing
from time import time



MEMORY_LIMIT = 10000	# set to None if you don't want to apply any limit

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


def get_word_vectors(model, words):
	vectors = []

	for w in words:
		try:
			vectors.append(model[w])
		except KeyError:
			continue
		except Exception as e:
			# should log this event
			continue

	return vectors if vectors else None


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
df = pp.text_preprocessing(df, 'body', 'title')

corpus = df['body'].tolist()

model = Word2Vec(	corpus, 
					min_count=1,
					window=5,
					size=300,
					sample=6e-5, 
					alpha=0.03, 
					min_alpha=0.0007, 
					negative=20,
					workers=multiprocessing.cpu_count()-1)

print(model.wv.most_similar(positive=['enhancement']))



# labels = pd.Series(pp.get_unique_values(df, 'labels').keys().values)

# model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)


# l_dict = {}
# for data_l in labels.values:
# 	best_match = None
# 	max_similarity = -math.inf

# 	try:
# 		l_vector = get_phrase_embedding(model, data_l)
		
# 		for target_l in LABELS:
# 			target_vector = get_phrase_embedding(model, target_l)

# 			cur_similarity = similarity(l_vector, target_vector)

# 			if cur_similarity > max_similarity:
# 				max_similarity = cur_similarity
# 				best_match = target_l


# 		# print(f'{data_l}\n{best_match}\n{max_similarity}\n\n\n')
# 		l_dict[data_l] = best_match
# 	except:
# 		continue



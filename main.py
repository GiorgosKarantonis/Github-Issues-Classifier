import numpy as np
import pandas as pd



class MemoryLimit:
	def __init__(self):
		self.apply = True
		self.value = 20


MEMORY_LIMIT = MemoryLimit()



def fetch_data(base_url='https://storage.googleapis.com/codenet/issue_labels/00000000000'):
	# download data
	df = pd.concat([pd.read_csv(f'{base_url}{i}.csv.gz') for i in range(10)])
	
	# clean 'url' column
	df['url'] = df['url'].str.replace('"', '')

	# drop redundant columns
	df = drop_columns(df, 'num_labels', 'c_bug', 'c_feature', 'c_question', 'class_int')

	# clean 'labels' column
	df['labels'] = clean_labels(df['labels'].values)

	# save cleaned dataframe
	df.to_pickle('data/github.pkl')

	return df


def load_github_data(fetch=False, base_url='https://storage.googleapis.com/codenet/issue_labels/00000000000'):
	return fetch_data() if fetch else pd.read_pickle('data/github.pkl')


def drop_columns(df, *columns):
	for c in columns:
		df = df.drop(c, axis=1)

	return df


def clean_labels(labels):
	labels_fixed = []

	for i, label_set in enumerate(labels):
		label_set = label_set.replace('[', '').replace(']', '').replace('"', '').split(', ')

		cur_labels = []
		for label in label_set:
			cur_labels.append(label)
		
		labels_fixed.append(cur_labels)

	return pd.Series(labels_fixed)


def min_presence(df, feature='labels', p=.001):
	thresh = int(p * len(df))
	
	features_count = df.explode(feature)[feature].value_counts()
	n_features = features_count.where(features_count.values >= thresh).dropna()
	
	return n_features.keys().values, n_features.values


def assign(s, values):
	labels_df = pd.DataFrame(np.zeros((len(s), len(values))), columns=values)

	for i, label_set in enumerate(s):
		if i < MEMORY_LIMIT.value:
			for l in label_set:
				labels_df.iloc[i][l] = 1
		else:
			break

	return labels_df


def wrap_up(df, **kwargs):
	df = pd.concat([github_df, pd.concat(kwargs['to_add'], axis=1)], axis=1)
	df = drop_columns(df, kwargs['to_drop'])

	df.to_pickle('data/github_final.pkl')

	return df



# load dataset or pass 'fetch=True' to download from scratch
github_df = load_github_data()[:MEMORY_LIMIT.value] if MEMORY_LIMIT.apply else load_github_data()


# filter out rare classes
labels, _ = min_presence(github_df, p=.1)

# vectorize labels
labels_df = assign(github_df['labels'], labels)


# start NLP
title_df = github_df['title']
body_df = github_df['body']


# finish preprocessing
github_df = wrap_up(github_df, to_add=[title_df, body_df, labels_df], to_drop=['title', 'body', 'labels'])


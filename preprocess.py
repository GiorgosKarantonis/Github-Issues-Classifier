import numpy as np
import pandas as pd

from string import punctuation

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

import nltk
nltk.download('stopwords')
nltk.download('wordnet')



# define the hyperparameters
FETCH = True
MEMORY_LIMIT = None



def download_data(memory_limit=None, save_dataset=True, base_url='https://storage.googleapis.com/codenet/issue_labels/00000000000'):
	print('\nDownloading Dataset...')

	df = pd.concat([pd.read_csv(f'{base_url}{i}.csv.gz') for i in range(10)])[:memory_limit]

	if save_dataset:
		df.to_pickle('data/github_raw.pkl')

	return df
	

def drop_columns(df, *columns):
	print('\nRemoving Reduntant Columns...')

	for c in columns:
		try:
			df = df.drop(c, axis=1)
		except KeyError:
			print(c)
			pass

	return df


def clean_labels(labels):
	print('\nCleaning Labels...')

	labels_fixed = []

	for i, label_set in enumerate(labels):
		# convert labels from string to list
		label_set = label_set.lower().replace('[', '').replace(']', '').replace('"', '').split(', ')

		# keep only unique labels
		label_set = set(label_set)

		cur_labels = []
		for label in label_set:
			cur_labels.append(label)

		labels_fixed.append(cur_labels)

	return pd.Series(labels_fixed)


def get_reference_info(df):
	print('\nGetting Issue Info...')

	reference_df = df['url'].str.extract(r'.*github\.com/(?P<user>.+)/(?P<repo_name>.+)/issues/(?P<issue_number>\d+)')

	df = transform(df, to_add=[reference_df[feature] for feature in reference_df])

	return df


def min_presence(df, feature='labels', p=.001):
	print('\nFiltering out Reduntant Labels...')

	thresh = int(p * len(df))
	
	features_count = df.explode(feature)[feature].value_counts()
	n_features = features_count.where(features_count.values >= thresh).dropna()
	
	return n_features.keys().values, n_features.values


def vectorize(s, values, prefix=None):
	print('\nVectorizing Features...')

	series_length = len(s)
	labels_df = pd.DataFrame(np.zeros((len(s), len(values))), columns=values)
	
	for i, label_set in enumerate(s):
		print(f'{i+1} / {series_length}\r', end='')
		
		for l in label_set:
			labels_df.iloc[i][l] = 1

	if prefix:
		labels_df.columns = [f'{prefix}_{v}' for v in values]

	return labels_df


def text_preprocessing(df, *text_features, lemmatize=False, stem=True):
	print('\nProcessing Text Data...')

	n_features = len(text_features)

	for i, feature in enumerate(text_features):
		print(f'\tConverting to Lowercase Feature: {i+1} / {n_features}\r', end='')
		df[feature] = df[feature].str.lower()

		# df[feature] = df[feature].str.decode('unicode_escape')

		print(f'\tTokenizing Feature: {i+1} / {n_features}\r', end='')
		df[feature] = df[feature].apply(word_tokenize)

		print(f'\tRemoving Stopwords Feature: {i+1} / {n_features}\r', end='')
		df[feature] = [[word for word in issue if word not in stopwords.words('english') and word not in punctuation] for issue in df[feature]]
		
		if lemmatize:
			print(f'\tLemmatizing Feature: {i+1} / {n_features}\r', end='')
			df[feature] = [' '.join([WordNetLemmatizer().lemmatize(word) for word in issue]) for issue in df[feature]]
		
		if stem:
			print(f'\tStemming Feature: {i+1} / {n_features}\r', end='')
			df[feature] = [' '.join([PorterStemmer().stem(word) for word in issue]) for issue in df[feature]]

	return df


def transform(df, **kwargs):
	print('\nTransforming DataFrame...')

	try:
		for feature in kwargs['to_add']:
			df.reset_index(drop=True, inplace=True)
			feature.reset_index(drop=True, inplace=True)

			df = pd.concat([df, feature], axis=1)
	except KeyError:
		pass

	try:
		df = drop_columns(df, kwargs['to_drop'])
	except KeyError:
		pass

	return df


def preprocess(df):
	# clean 'url' column
	df['url'] = df['url'].str.replace('"', '')

	# get issue reference data
	df = get_reference_info(df)

	# drop redundant columns
	df = drop_columns(df, 'num_labels', 'c_bug', 'c_feature', 'c_question', 'class_int')

	# clean 'labels' column
	df['labels'] = clean_labels(df['labels'].values)

	# filter out rare classes
	labels, _ = min_presence(df, p=.01)

	# vectorize labels
	labels_df = vectorize(df['labels'], labels, prefix='label')

	# add the vectorized labels
	df = transform(df, to_add=[labels_df])

	# normalize the textual features
	df = text_preprocessing(df, 'title', 'body')

	return df


def fetch_data(look_for_downloaded=True, memory_limit=None, base_url='https://storage.googleapis.com/codenet/issue_labels/00000000000'):
	if look_for_downloaded:
		try:
			df = pd.read_pickle('data/github_raw.pkl')[:memory_limit]
		except:
			df = download_data(memory_limit=memory_limit, base_url=base_url)[:memory_limit]
	else:
		df = download_data(memory_limit=memory_limit, base_url=base_url)[:memory_limit]

	# process the raw dataframe
	df = preprocess(df)

	# save cleaned dataframe
	df.to_pickle('data/github.pkl')

	return df


def load_github_data(fetch=False, memory_limit=None, base_url='https://storage.googleapis.com/codenet/issue_labels/00000000000'):
	return fetch_data(memory_limit=memory_limit, base_url=base_url) if fetch else pd.read_pickle('data/github.pkl')[:memory_limit]



if __name__ == '__main__':
	# load the preprocessed dataset or set 'FETCH=True' to download from scratch
	github_df = load_github_data(fetch=FETCH, memory_limit=MEMORY_LIMIT)

import numpy as np
import pandas as pd

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification



def download_data(memory_limit=None, save=True, base_url='https://storage.googleapis.com/codenet/issue_labels/00000000000'):
	print('Downloading Dataset...\n')

	df = pd.concat([pd.read_csv(f'{base_url}{i}.csv.gz') for i in range(10)])[:memory_limit]

	if save:
		df.to_pickle('data/github_raw.pkl')

	return df
	

def drop_columns(df, *columns):
	print('Removing Reduntant Columns...\n')

	for c in columns:
		try:
			df = df.drop(c, axis=1)
		except KeyError:
			print(c)
			pass

	return df


def clean_labels(labels):
	print('Cleaning Labels...\n')

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
	print('Getting Issue Info...\n')

	reference_df = df['url'].str.extract(r'.*github\.com/(?P<user>.+)/(?P<repo_name>.+)/issues/(?P<issue_number>\d+)')

	df = transform(df, to_add=[reference_df[feature] for feature in reference_df])

	return df


def get_unique_values(df, feature):
	return df.explode(feature)[feature].value_counts()


def min_presence(df, feature='labels', p=.001):
	print('Filtering out Reduntant Labels...\n')

	thresh = int(p * len(df))
	
	features_count = get_unique_values(df, feature)
	n_features = features_count.where(features_count.values >= thresh).dropna()
	
	return n_features.keys().values, n_features.values


def vectorize(s, values, prefix=None):
	print('Vectorizing Features...\n')

	series_length = len(s)
	labels_df = pd.DataFrame(np.zeros((len(s), len(values))), columns=values)
	
	for i, label_set in enumerate(s):
		print(f'{i+1} / {series_length}\r', end='')
		
		for l in label_set:
			labels_df.iloc[i][l] = 1

	if prefix:
		labels_df.columns = [f'{prefix}_{v}' for v in values]

	return labels_df


def clean_text_data(df, *features):
	for feature in features:
		df[feature] = df[feature].str.replace(r'\\r', '').str.lower()

	return df


def text_preprocessing(df, *features):
	print('Processing Text Data...\n')

	for i, feature in enumerate(features):
		# tokenize
		try:
			df[feature] = df[feature].apply(word_tokenize)
		except LookupError:
			nltk.download('punkt')
			df[feature] = df[feature].apply(word_tokenize)

		# remove stop words and noise
		try:
			df[feature] = [[word for word in issue if word not in stopwords.words('english') and word.isalpha()] for issue in df[feature]]
		except LookupError:
			nltk.download('stopwords')
			df[feature] = [[word for word in issue if word not in stopwords.words('english') and word.isalpha()] for issue in df[feature]]

		# stem the tokens
		try:
			df[feature] = [[PorterStemmer().stem(word) for word in issue] for issue in df[feature]]
		except LookupError:
			nltk.download('wordnet')
			df[feature] = [[PorterStemmer().stem(word) for word in issue] for issue in df[feature]]

	return df


def make_combinations(target, observed):
	combinations = []
	
	for t in target:
	    for o in observed:
	        combinations.append([t, o])

	return combinations


def check_paraphrase(inputs):
	tokenizer = BertTokenizer.from_pretrained('bert-base-cased-finetuned-mrpc')
	model = TFBertForSequenceClassification.from_pretrained('bert-base-cased-finetuned-mrpc')

	inputs = tokenizer(	inputs, 
						padding=True, 
						truncation=True, 
						return_tensors='tf')

	logits = model(inputs)[0]
	outputs = tf.nn.softmax(logits, axis=1).numpy()

	not_paraphrase_likelihood = outputs[:, 0]
	paraphrase_likelihood = outputs[:, 1]

	return not_paraphrase_likelihood, paraphrase_likelihood


def disambiguate_labels(labels_dict, disambiguate='keep_most_probable'):
	# can use threshold here instead of the next cell
    assert disambiguate in ['keep_most_probable', 'drop_all']

    cleaned_dict = {}

    for key in labels_dict:
        if len(labels_dict[key]) > 1:
            if disambiguate == 'drop_all':
                cleaned_dict[key] = None
            else:
                max_likelihood = 0
                best_match = None

                for label, likelihood in labels_dict[key]:
                    if likelihood > max_likelihood:
                        max_likelihood = likelihood
                        best_match = label
                
                cleaned_dict[key] = best_match
        else:
            label, likelihood = labels_dict[key][0]
            cleaned_dict[key] = label
    
    return cleaned_dict


def map_labels(label_series, mapping):
	for i, label_list in enumerate(label_series):
	    
	    temp_labels = []
	    for l in label_list:
	        try:
	            temp_labels.append(mapping[l])
	        except:
	            pass
	    
	    if temp_labels:
	        mapped_labels.append(temp_labels)
	    else:
	        mapped_labels.append(['undefined'])


	return pd.Series(mapped_labels)


def transform(df, **kwargs):
	print('Transforming DataFrame...\n')

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


def preprocess(df, save=True):
	# clean 'url' column
	df['url'] = df['url'].str.replace('"', '')

	# get issue reference data
	df = get_reference_info(df)

	# drop redundant columns
	df = drop_columns(df, 'repo', 'num_labels', 'c_bug', 'c_feature', 'c_question', 'class_int')

	# clean 'labels' column
	df['labels'] = clean_labels(df['labels'].values)

	# filter out rare classes
	labels, _ = min_presence(df, p=.01)

	# vectorize labels
	labels_df = vectorize(df['labels'], labels, prefix='label')

	df = clean_text_data(df, 'title', 'body')

	# add the vectorized labels
	df = transform(df, to_add=[labels_df])

	if save:
		# save cleaned dataframe
		df.to_pickle('data/github.pkl')

	return df


def fetch_github_data(look_for_downloaded=True, memory_limit=None, base_url='https://storage.googleapis.com/codenet/issue_labels/00000000000'):
	if look_for_downloaded:
		try:
			df = pd.read_pickle('data/github_raw.pkl')[:memory_limit]
		except:
			df = download_data(memory_limit=memory_limit, base_url=base_url)[:memory_limit]
	else:
		df = download_data(memory_limit=memory_limit, base_url=base_url)[:memory_limit]

	return df


def load_data(fetch=False, memory_limit=None, file='data/github.pkl', base_url='https://storage.googleapis.com/codenet/issue_labels/00000000000'):
	return fetch_github_data(memory_limit=memory_limit, base_url=base_url) if fetch else pd.read_pickle(file)[:memory_limit]



if __name__ == '__main__':
	# define the hyperparameters
	FETCH = True
	MEMORY_LIMIT = 100
	SAVE = True


	# load the preprocessed dataset or set 'FETCH=True' to download from scratch
	df = load_data(fetch=FETCH, memory_limit=MEMORY_LIMIT)

	df['url'] = df['url'].str.replace('"', '')
	df = get_reference_info(df)
	df = drop_columns(df, 'repo', 'num_labels', 'c_bug', 'c_feature', 'c_question', 'class_int')
	df['labels'] = clean_labels(df['labels'].values)

	# filter out rare classes
	labels, _ = min_presence(df, p=.01)


	unique_labels = get_unique_values(df, 'labels').keys().values
	
	paraphrase_list = make_combinations(LABELS, unique_labels)
	_, paraphrase_likelihood = check_paraphrase(paraphrase_list)


	label_mapping = {}
	for i, pair in enumerate(paraphrase_list):
	    if paraphrase_likelihood[i] > .5:
	        target_l, real_l = pair[0], pair[1]
	        try:
	            label_mapping[real_l].append((target_l, paraphrase_likelihood[i]))
	        except:
	            label_mapping[real_l] = []
	            label_mapping[real_l].append((target_l, paraphrase_likelihood[i]))
	
	label_mapping = disambiguate_labels(label_mapping)

	df['labels'] = map_labels(df['labels'], label_mapping)


	df = clean_text_data(df, 'title', 'body')

	if 'undefined' not in LABELS:
	    LABELS.append('undefined')

	labels_vectorized = vectorize(df['labels'], LABELS, prefix='label')
	df = pp.transform(df, to_add=[labels_vectorized])

	if SAVE:
		# save cleaned dataframe
		df.to_pickle('data/github.pkl')


	



import pandas as pd



def load_github_data(fetch=False, base_url='https://storage.googleapis.com/codenet/issue_labels/00000000000'):
	if fetch:
		# download data
		df = pd.concat([pd.read_csv(f'{base_url}{i}.csv.gz') for i in range(10)])
		
		# clean 'url' column
		github_df['url'] = github_df['url'].str.replace('"', '')

		# drop redundant columns
		github_df = drop_columns(github_df, 'num_labels', 'c_bug', 'c_feature', 'c_question', 'class_int')

		# clean 'labels' column
		github_df['labels'] = clean_labels(github_df['labels'].values)

		# save cleaned dataframe
		github_df.to_pickle('data/github.pkl')
	else:
		df = pd.read_pickle('data/github.pkl')

	return df


def drop_columns(df, *colums):
	for c in colums:
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


def get_unique_values(df, feature):
	return pd.unique(df.explode(feature)[feature])


def keep_n_features(df, feature='labels', n=.001):
	min_presence = int(n * len(df))
	
	features_count = df.explode(feature)[feature].value_counts()
	n_features = features_count.where(features_count.values >= min_presence).dropna()
	
	return n_features.keys().values, n_features.values



# load dataset
# pass 'fetch=True' to download from scratch
github_df = load_github_data()

labels, _ = keep_n_features(github_df)
print(labels)








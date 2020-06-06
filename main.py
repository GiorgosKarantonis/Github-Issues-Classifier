import numpy as np
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
	




# load or download raw dataset
github_df = load_github_data()

unique_labels = get_unique_values(github_df, 'labels')

print(unique_labels.shape)



'''
############################################

test = github_df.explode('labels')['labels'].value_counts()

counter = 0
min_count = 1000

for t in test.values:
	if t > min_count:
		counter += 1

print(len(github_df))
print(max(test.values))
print()
print(f'{len(test.values)} ---> {counter}')

############################################


labels_df = pd.DataFrame()

for label in unique_labels[:10]:
	labels_df[label] = pd.Series(np.zeros(len(github_df)))


print(labels_df.shape)
print(labels_df.columns.values)
'''



# print(github_df.columns.values)








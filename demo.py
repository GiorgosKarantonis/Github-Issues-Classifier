import numpy as np
import pandas as pd

from preprocess import text_preprocessing



MEMORY_LIMIT = 50



df_raw = pd.read_pickle('data/github_raw.pkl')
df_raw_small = df_raw.head(MEMORY_LIMIT)


df = pd.read_pickle('data/github.pkl')
df_small = df.head(MEMORY_LIMIT)
df_small = text_preprocessing(df_small, 'title', 'body', lemmatize=False, stem=True)


labels = df.explode('labels')['labels'].value_counts()
labels_small = df_small.explode('labels')['labels'].value_counts()



print('\n\n\nTOTAL NUMBER OF EXAMPLES\n')
print(len(df))


print('\n\n\nORIGINAL FEATURES\n')
for c in df_raw.columns:
	print(c)


print('\n\n\nFEATURES AFTER PREPROCESSING\n')
for c in df.columns:
	print(c)


print('\n\n\nNUMBER OF LABELS WITHOUT FILTERING\n')
print(len(labels))


print('\n\n\n10 MOST COMMON LABELS\n')
print(labels[:10])


print('\n\n\n10 LEAST COMMON LABELS\n')
print(labels[-10:])


print('\n\n\nSAMPLE TITLE BEFORE PREPROCESSING\n')
print(f'{df_raw_small.title.head(1).values}')

print('\n\n\nSAMPLE TITLE AFTER PREPROCESSING\n')
print(f'{df_small.title.head(1).values}')


print('\n\n\nSAMPLE BODY BEFORE PREPROCESSING\n')
print(f'{df_raw_small.body.head(1).values}')

print('\n\n\nSAMPLE BODY AFTER PREPROCESSING\n')
print(f'{df_small.body.head(1).values}')
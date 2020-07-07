import numpy as np
import pandas as pd
import tensorflow as tf

import preprocessing as pp

from transformers import DistilBertConfig
from transformers import DistilBertTokenizer, DistilBertTokenizerFast
from transformers import TFDistilBertModel, TFDistilBertForSequenceClassification

from transformers import BertTokenizer, TFBertForSequenceClassification

import time
import itertools



MEMORY_LIMIT = 100	# set to None if you don't want to apply any limit

EXAMPLE_LABELS = [
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

LABELS = [
	'bug', 
	'enhancement', 
	'question'
]



def get_paraphrased_values(inputs):
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



df = pp.load_data(memory_limit=MEMORY_LIMIT)
# labels = [df[c] for c in df if c.startswith('label_')]

unique_labels = pp.get_unique_values(df, 'labels').keys().values
combinations = np.array(list(itertools.combinations(unique_labels, 2))).tolist()
_, paraphrase_likelihood = get_paraphrased_values(combinations)

for i, pair in enumerate(combinations):
	if paraphrase_likelihood[i] > .8:
		print(f'{pair[0]}\t{pair[1]}\t\t{paraphrase_likelihood[i]}')


# for c in df.columns:
# 	if c.startswith('label_') and c.strip('label_') not in LABELS:
# 		df = df.drop(c, axis=1)


# start = time.time()

# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')


# # inputs_title = tokenizer(	df['title'].values.tolist(), 
# # 							padding=True, 
# # 							truncation=True, 
# # 							return_tensors='tf')

# inputs_body = tokenizer(	df['body'].values.tolist(), 
# 							padding=True, 
# 							truncation=True, 
# 							return_tensors='tf')

# # inputs_body['labels'] = tf.transpose(tf.convert_to_tensor(labels))
# inputs_body['labels'] = tf.convert_to_tensor(np.array(labels).reshape(-1, 1))


# outputs = model(inputs_body)

# print(outputs)

# loss, logits = outputs[:2]

# print(logits.shape)
# print(loss)

# print(f'Total time elapsed: {time.time() - start}')


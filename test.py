import pandas as pd
import numpy as np

import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel

import preprocessing as pp

import time



MEMORY_LIMIT = 1000	# set to None if you don't want to apply any limit

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



df = pp.load_data(memory_limit=MEMORY_LIMIT)
# print(pp.get_unique_values(df, 'labels'))


start = time.time()

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')


inputs_title = tokenizer(	df['title'].values.tolist(), 
							padding=True, 
							truncation=True, 
							return_tensors='tf')

inputs_body = tokenizer(	df['body'].values.tolist(), 
							padding=True, 
							truncation=True, 
							return_tensors='tf')

outputs = model(inputs)
last_hidden_states = outputs[0]

print(last_hidden_states.shape)
print(f'Total time elapsed: {time.time() - start}')


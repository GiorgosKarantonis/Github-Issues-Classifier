import numpy as np
import pandas as pd
import tensorflow as tf

import preprocessing as pp

from transformers import DistilBertConfig
from transformers import DistilBertTokenizer, DistilBertTokenizerFast
from transformers import TFDistilBertModel, TFDistilBertForSequenceClassification

import time



MEMORY_LIMIT = 100	# set to None if you don't want to apply any limit

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
labels = [df[c] for c in df if c.startswith('label_')]



# Detect Similar labels
tokenizer = BertTokenizer.from_pretrained('bert-base-cased-finetuned-mrpc')
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased-finetuned-mrpc')

classes = ["not paraphrase", "is paraphrase"]

l_1 = "bug"
l_2 = "enhancement"
l_3 = "type: bug"

paraphrase = tokenizer(l_1, l_3, return_tensors="tf")
not_paraphrase = tokenizer(l_1, l_2, return_tensors="tf")

paraphrase_classification_logits = model(paraphrase)[0]
not_paraphrase_classification_logits = model(not_paraphrase)[0]

paraphrase_results = tf.nn.softmax(paraphrase_classification_logits, axis=1).numpy()[0]
not_paraphrase_results = tf.nn.softmax(not_paraphrase_classification_logits, axis=1).numpy()[0]

# Should be paraphrase
for i in range(len(classes)):
	print(f"{classes[i]}: {paraphrase_results[i]}")
print()
# Should not be paraphrase
for i in range(len(classes)):
	print(f"{classes[i]}: {not_paraphrase_results[i]}")




# start = time.time()

# # Check here for multi-label classification
# # https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d
# # https://github.com/huggingface/transformers/issues/1465
# # https://towardsdatascience.com/multi-label-classification-using-bert-roberta-xlnet-xlm-and-distilbert-with-simple-transformers-b3e0cda12ce5


# configuration = DistilBertConfig(num_labels=len(labels))

# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', config=configuration)


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


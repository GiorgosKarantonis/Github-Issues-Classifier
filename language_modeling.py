import numpy as np
import pandas as pd

import tensorflow as tf

from transformers import BertTokenizer, DistilBertTokenizer, DistilBertTokenizerFast
from transformers import TFBertModel, TFDistilBertModel
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# from preprocessing import get_n_chunks



tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer_dstl = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer_dstl_fast = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

model = TFBertModel.from_pretrained('bert-base-cased')
model_dstl = TFDistilBertModel.from_pretrained('distilbert-base-uncased')



def get_n_chunks(df, chunk_size):
	df_size = len(df)

	return int(df_size / chunk_size) if df_size % chunk_size == 0 else int(df_size / chunk_size) + 1


def get_embeddings(data, tokenizer=tokenizer_dstl, model=model_dstl):
    inputs = tokenizer( data, 
                        padding=True, 
                        truncation=True, 
                        return_tensors='tf')

    outputs = model(inputs)
    logits = outputs[0]

    return logits


def summarize(text):
	tokenizer = T5Tokenizer.from_pretrained('t5-small')
	model = TFT5ForConditionalGeneration.from_pretrained('t5-small')
	
	inputs = tokenizer.encode(text, return_tensors="tf")  # Batch size 1
	
	return model.generate(inputs)


def refactor_text(*text_features, labels, threshold=512, equal_size=True):
    raise NotImplementedError('Use summarization to deal with large sequences. ')


def save_tensor(tensor, path):
    array = tensor.numpy()
    
    np.save(path, array)


def load_array(path, to_tensor=True):
    array = np.load(f'{path}.npy')
    
    return tf.convert_to_tensor(array, dtype=array.dtype) if to_tensor else array



if __name__=='__main__':
    pass
    # to do: fix truncation issue
    # embeddings_title = get_embeddings(df['title'].values.tolist())  # n_examples x n_words (or threshold) x 768
    # embeddings_body = get_embeddings(df['body'].values.tolist())  # n_examples x n_words (or threshold) x 768

    # labels = np.transpose([df[c] for c in df.columns if c.startswith('label_')])
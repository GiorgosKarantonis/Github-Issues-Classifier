import numpy as np
import pandas as pd

import tensorflow as tf

from transformers import DistilBertTokenizer, TFDistilBertModel



tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertModel.from_pretrained('distilbert-base-uncased', training=False)



def get_embeddings(data, tokenizer=tokenizer, model=model):
    inputs = tokenizer( data, 
                        padding=True, 
                        truncation=True, 
                        return_tensors='tf')

    outputs = model(inputs)
    logits = outputs[0]

    return logits


def save_tensor(tensor, path):
    array = tensor.numpy()
    
    np.save(path, array)


def load_array(path, to_tensor=True):
    array = np.load(f'{path}.npy')
    
    return tf.convert_to_tensor(array, dtype=array.dtype) if to_tensor else array



if __name__=='__main__':
    pass
    # embeddings_title = get_embeddings(df['title'].values.tolist())  # n_examples x n_words (or threshold) x 768
    # embeddings_body = get_embeddings(df['body'].values.tolist())  # n_examples x n_words (or threshold) x 768

    # labels = np.transpose([df[c] for c in df.columns if c.startswith('label_')])
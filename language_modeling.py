import numpy as np
import pandas as pd

import tensorflow as tf

from transformers import BertTokenizer, DistilBertTokenizer, DistilBertTokenizerFast
from transformers import TFBertModel, TFDistilBertModel

from preprocessing import get_n_chunks



tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer_dstl = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer_dstl_fast = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

model = TFBertModel.from_pretrained('bert-base-cased')
model_dstl = TFDistilBertModel.from_pretrained('distilbert-base-uncased')



def get_embeddings(data, tokenizer=tokenizer_dstl, model=model_dstl):
    inputs = tokenizer( data, 
                        padding=True, 
                        truncation=True, 
                        return_tensors='tf')

    outputs = model(inputs)
    logits = outputs[0]

    return logits


def refactor_text(text_list, labels, threshold=512, equal_size=True):
    new_text, new_labels = [], []

    for row, l in zip(text_list, labels):
        if len(row) > threshold:
            n_chunks = get_n_chunks(row, threshold)
            
            for i in range(n_chunks):
                end = (i+1)*threshold
                
                if equal_size and len(row) - end < threshold:
                    new_row = row[-threshold:]
                else:
                    new_row = row[i*threshold:end]
                                    
                new_text.append(new_row)
                new_labels.append(l)
        else:
            new_text.append(row)
            new_labels.append(l)


    return new_text, new_labels


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
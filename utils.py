import json 
import pandas as pd



def load_models_meta():
    with open('models.json') as json_file: 
        models_meta = json.load(json_file)


    return models_meta


def get_n_chunks(df, chunk_size):
    df_size = len(df)

    return int(df_size / chunk_size) if df_size % chunk_size == 0 else int(df_size / chunk_size) + 1


def get_unique_values(df, feature):
    return df.explode(feature)[feature].value_counts()


def get_pretrained_model(meta):    
    if meta['name'] == 'gpt2':
        from transformers import GPT2Tokenizer, TFGPT2Model
        
        tokenizer = GPT2Tokenizer.from_pretrained(meta['model'])
        model = TFGPT2Model.from_pretrained(meta['model'])
        tokenizer.pad_token=tokenizer.eos_token
    elif meta['name'] == 'electra':
        from transformers import ElectraTokenizer, TFElectraModel
        
        tokenizer = ElectraTokenizer.from_pretrained(meta['model'])
        model = TFElectraModel.from_pretrained(meta['model'])
    elif meta['name'] == 'distilbert':
        from transformers import DistilBertTokenizer, TFDistilBertModel
        
        tokenizer = DistilBertTokenizer.from_pretrained(meta['model'])
        model = TFDistilBertModel.from_pretrained(meta['model'])
    elif meta['name'] == 'bert':
        from transformers import BertTokenizer, TFBertModel
        
        tokenizer = BertTokenizer.from_pretrained(meta['model'])
        model = TFBertModel.from_pretrained(meta['model'])
    else:
        raise NotImplementedError('Invalid model name...')
        
    
    return tokenizer, model
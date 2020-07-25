import os
import pickle
from time import time

import pandas as pd

import torch
from transformers import BartTokenizer, BartForConditionalGeneration



device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-6-6')
model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-6-6').to(device)



def update_logs(message):
    print(message)
    with open('./logs.txt', 'a') as f:
        f.write(f'{message}\n')


def summarize(	df, 
				feature, 
				tokenizer=tokenizer, 
				model=model,  
				checkpoint_every=10):

	summaries = []
	n_examples = len(df)

	chunk_size = 100
	n_chunks = int(n_examples / chunk_size) if n_examples % chunk_size == 0 else int(n_examples / chunk_size) + 1

	checkpoint_every = checkpoint_every


	start = time()
	with torch.no_grad():
	    for i in range(n_chunks):
	        start_i = time()
	        update_logs(f'Batch: {i+1}/{n_chunks}')

	        input_ids = tokenizer.batch_encode_plus(    df.iloc[i*chunk_size:(i+1)*chunk_size][feature], 
	                                                    padding=True, 
	                                                    truncation=True, 
	                                                    max_length=1024, 
	                                                    return_tensors='pt')['input_ids'].to(device)

	        summary_ids = model.generate(input_ids, max_length=512)

	        for i_id, s_id in enumerate(summary_ids):
	            summaries.append([df.iloc[i*chunk_size + i_id]['index'], 
	                              tokenizer.decode( s_id, 
	                                                skip_special_tokens=True, 
	                                                clean_up_tokenization_spaces=True)])

	        del input_ids, summary_ids
	        if device == 'cuda':
	        	torch.cuda.empty_cache()


	        if (i+1)%checkpoint_every == 0 or i+1 == n_examples:
	            update_logs('Creating checkpoint...')
	            with open(f'./checkpoint_{feature}_{i+1}.pkl', 'wb') as pkl:
	                pickle.dump(summaries, pkl)

	            summaries = []

	        update_logs(f'Time elapsed for batch {i+1}: {(time()-start_i)}sec\n')

	update_logs(f'\nTotal time elapsed: {(time()-start)/60}min')



if __name__ == '__main__':
	df = pd.read_pickle('http://cs-people.bu.edu/giorgos/labelbot/long_bodies.pkl')
	summarize(df, 'body')

	df = pd.read_pickle('http://cs-people.bu.edu/giorgos/labelbot/long_titles.pkl')
	summarize(df, 'title')





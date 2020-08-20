# Github-Issues-Classifier
# Copyright(C) 2018, 2019, 2020 Georgios (Giorgos) Karantonis
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os
import click

import numpy as np
import pandas as pd

import paraphrase_detector



def download_data(memory_limit=None, save=True, base_url='https://storage.googleapis.com/codenet/issue_labels/00000000000'):
    print('Downloading Dataset...\n')

    df = pd.concat([pd.read_csv(f'{base_url}{i}.csv.gz') for i in range(10)])[:memory_limit]

    if save:
        df.to_pickle('data/github_raw.pkl')

    return df


def drop_columns(df, *columns):
    print('Removing Redundant Columns...\n')

    for c in columns:
        try:
            df = df.drop(c, axis=1)
        except KeyError:
            print(c)
            pass

    return df


def clean_labels(labels):
    print('Cleaning Labels...\n')

    labels_fixed = []

    for i, label_set in enumerate(labels):
        # convert labels from string to list
        label_set = label_set.lower().replace('[', '').replace(']', '').replace('"', '').split(', ')

        # keep only unique labels
        label_set = set(label_set)

        cur_labels = []
        for label in label_set:
            cur_labels.append(label)

        labels_fixed.append(cur_labels)

    return pd.Series(labels_fixed)


def get_reference_info(df):
    print('Getting Issue Info...\n')

    reference_df = df['url'].str.extract(r'.*github\.com/(?P<user>.+)/(?P<repo_name>.+)/issues/(?P<issue_number>\d+)')

    df = transform(df, to_add=[reference_df[feature] for feature in reference_df])

    return df


def min_presence(df, feature='labels', p=.001):
    print('Filtering out Redundant Labels...\n')

    thresh = int(p * len(df))
    
    features_count = get_unique_values(df, feature)
    n_features = features_count.where(features_count.values >= thresh).dropna()
    
    return n_features.keys().values, n_features.values


def vectorize(s, values, prefix=None):
    print('Vectorizing Features...\n')

    series_length = len(s)
    labels_df = pd.DataFrame(np.zeros((len(s), len(values))), columns=values)
    
    for i, label_set in enumerate(s):
        print(f'{i+1} / {series_length}\r', end='')
        
        for l in label_set:
            labels_df.iloc[i][l] = 1

    if prefix:
        labels_df.columns = [f'{prefix}_{v}' for v in values]

    return labels_df


def clean_text_data(df, *features):
    print('Cleaning Text Data...\n')

    for feature in features:
        df[feature] = df[feature].str.replace(r'\\r', '').str.lower()
        df[feature] = df[feature].str.split().str.join(' ')

    return df


def transform(df, **kwargs):
    print('Transforming DataFrame...\n')

    try:
        for feature in kwargs['to_add']:
            df.reset_index(drop=True, inplace=True)
            feature.reset_index(drop=True, inplace=True)

            df = pd.concat([df, feature], axis=1)
    except KeyError:
        pass

    try:
        df = drop_columns(df, kwargs['to_drop'])
    except KeyError:
        pass

    return df


def preprocess(df, save=True, save_to='data/github.pkl'):
    df['url'] = df['url'].str.replace('"', '')
    
    df = get_reference_info(df)
    
    df = drop_columns(df, 'repo', 'num_labels', 'c_bug', 'c_feature', 'c_question', 'class_int')
    
    # df['labels'] = clean_labels(df['labels'].values)
    
    df = clean_text_data(df, 'title', 'body')

    df['labels'], LABELS = paraphrase_detector.main(df['labels'])

    labels_vectorized = vectorize(df['labels'], LABELS, prefix='label')
    df = transform(df, to_add=[labels_vectorized])

    if save:
      df.to_pickle(save_to)

    return df


def fetch_github_data(look_for_downloaded=True, memory_limit=None, base_url='https://storage.googleapis.com/codenet/issue_labels/00000000000'):
    if look_for_downloaded:
        try:
            df = pd.read_pickle('data/github_raw.pkl')[:memory_limit]
        except:
            df = download_data(memory_limit=memory_limit, base_url=base_url)[:memory_limit]
    else:
        df = download_data(memory_limit=memory_limit, base_url=base_url)[:memory_limit]

    return df


def load_data(fetch=False, memory_limit=None, file='data/github.pkl', base_url='https://storage.googleapis.com/codenet/issue_labels/00000000000'):
    return fetch_github_data(memory_limit=memory_limit, base_url=base_url) if fetch else pd.read_pickle(file)[:memory_limit]


@click.command()
@click.option('--fetch', '-F', default=False, type=bool)
@click.option('--limit', '-L', default=None, type=int)
def cli(fetch, limit):
    if not os.path.exists('data'):
        os.mkdir('data')

    df = load_data(fetch=fetch, memory_limit=limit)
    df = preprocess(df)



if __name__ == '__main__':
    cli()


    



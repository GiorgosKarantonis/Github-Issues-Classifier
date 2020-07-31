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



def split_to_classes(   df, 
                        to_keep=['title', 'body', 'label_bug', 'label_question', 'label_enhancement', 'label_undefined'], 
                        save=False, 
                        path='./'):
    bugs_df = df.query('label_bug == 1 and \
                        label_question == 0 and \
                        label_enhancement == 0')[to_keep]

    questions_df = df.query('label_bug == 0 and \
                             label_question == 1 and \
                             label_enhancement == 0')[to_keep]

    enhancements_df = df.query('label_bug == 0 and \
                                label_question == 0 and \
                                label_enhancement == 1')[to_keep]

    undefined_df = df.query('label_undefined == 1')[to_keep]

    combinations_df = df.query('(label_bug == 1 and label_question == 1 and label_enhancement == 0) or \
                                (label_bug == 1 and label_question == 0 and label_enhancement == 1) or \
                                (label_bug == 0 and label_question == 1 and label_enhancement == 1) or \
                                (label_bug == 1 and label_question == 1 and label_enhancement == 1)')[to_keep]

    if save:
        if not path.endswith('/'):
            path = f'{path}/'

        bugs_df.to_pickle(f'{path}bugs.pkl')
        questions_df.to_pickle(f'{path}questions.pkl')
        enhancements_df.to_pickle(f'{path}enhancements.pkl')
        undefined_df.to_pickle(f'{path}undefined.pkl')
        combinations_df.to_pickle(f'{path}combinations.pkl')


    return bugs_df, questions_df, enhancements_df, undefined_df, combinations_df



def get_labels_stats(df):
    total = len(df)

    # all the examples that are labelled
    # only as bugs
    b = len(df.query(   'label_bug == 1 and \
                        label_question == 0 and \
                        label_enhancement == 0 and \
                        label_undefined == 0'))

    # all the examples that are labelled
    # only as questions
    q = len(df.query(   'label_bug == 0 and \
                        label_question == 1 and \
                        label_enhancement == 0 and \
                        label_undefined == 0'))

    # all the examples that are labelled
    # only as enhancement
    e = len(df.query(   'label_bug == 0 and \
                        label_question == 0 and \
                        label_enhancement == 1 and \
                        label_undefined == 0'))

    # all the examples that are labelled
    # as undefined
    u = len(df.query(   'label_bug == 0 and \
                        label_question == 0 and \
                        label_enhancement == 0 and \
                        label_undefined == 1'))

    # all the examples that are labelled
    # only as both bug and question
    b_q = len(df.query('label_bug == 1 and \
                        label_question == 1 and \
                        label_enhancement == 0 and \
                        label_undefined == 0'))

    # all the examples that are labelled
    # only as both bug and enhancement
    b_e = len(df.query( 'label_bug == 1 and \
                        label_question == 0 and \
                        label_enhancement == 1 and \
                        label_undefined == 0'))

    # all the examples that are labelled
    # only as both question and enhancement
    q_e = len(df.query( 'label_bug == 0 and \
                        label_question == 1 and \
                        label_enhancement == 1 and \
                        label_undefined == 0'))

    # all the examples that are labelled
    # as bug, question and enhancement
    b_q_e = len(df.query(   'label_bug == 1 and \
                            label_question == 1 and \
                            label_enhancement == 1 and \
                            label_undefined == 0'))

    return pd.DataFrame([['Bug', b/total, b], 
                         ['Question', q/total, q], 
                         ['Enhancement', e/total, e], 
                         ['Undefined', u/total, u], 
                         ['Bug, Question', b_q/total, b_q], 
                         ['Bug, Enhancement', b_e/total, b_e], 
                         ['Question, Enhancement', q_e/total, q_e], 
                         ['Bug, Question, Enhancement', b_q_e/total, b_q_e], 
                         ['Total', total/total, total]], 
                        columns=['Labels Present', 'Fraction', 'Examples'])



def sample_df(df, n=None, frac=None):
    assert n != frac

    bugs_df, questions_df, enhancements_df, undefined_df, combinations_df = split_to_classes(df)

    if n:
        bugs_df = bugs_df.sample(n=n)
        questions_df = questions_df.sample(n=n)
        enhancements_df = enhancements_df.sample(n=n)
        undefined_df = undefined_df.sample(n=n)
    else:
        bugs_df = bugs_df.sample(frac=frac)
        questions_df = questions_df.sample(frac=frac)
        enhancements_df = enhancements_df.sample(frac=frac)
        undefined_df = undefined_df.sample(frac=frac)


    # shuffle the dataframes
    bugs_df = bugs_df.sample(frac=1)
    questions_df = questions_df.sample(frac=1)
    enhancements_df = enhancements_df.sample(frac=1)
    undefined_df = undefined_df.sample(frac=1)
    combinations_df = combinations_df.sample(frac=1)


    return bugs_df, questions_df, enhancements_df, undefined_df, combinations_df



def split_train_test(df, train_frac=.7, val_frac=.1, shuffle=True, save=True, path='./'):
    if shuffle:
        df = df.sample(frac=1)

    train_split = int(train_frac * (len(df)))

    train_df = df.iloc[:train_split]
    test_df = df.iloc[train_split:]


    val_split = int(val_frac * (len(train_df)))

    val_df = train_df.iloc[:val_split]
    val_df = train_df.iloc[val_split:]


    if save:
        if not path.endswith('/'):
            path = f'{path}/'

        train_df.to_pickle(f'{path}train.pkl')
        val_df.to_pickle(f'{path}val.pkl')
        test_df.to_pickle(f'{path}test.pkl')


    return train_df, val_df, test_df





import json
import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import label_ranking_average_precision_score as lrap
from sklearn.preprocessing import MultiLabelBinarizer

from matplotlib import pyplot as plt



def get_model_stats(y_true, model_outputs, b_thres=.5, q_thres=.5, e_thres=.5, plot_roc=True):
    b_scores, q_scores, e_scores = model_outputs[:, 0], model_outputs[:, 1], model_outputs[:, 2]
    b_true, q_true, e_true = y_true[:, 0], y_true[:, 1], y_true[:, 2]

    b_roc_auc = roc_auc_score(b_true, b_scores)
    q_roc_auc = roc_auc_score(q_true, q_scores)
    e_roc_auc = roc_auc_score(e_true, e_scores)

    b_fpr, b_tpr, _ = roc_curve(b_true, b_scores)
    q_fpr, q_tpr, _ = roc_curve(q_true, q_scores)
    e_fpr, e_tpr, _ = roc_curve(e_true, e_scores)

    if plot_roc:
        plt.figure()
        plt.title('ROC Curves \nfor Bugs, Questions and Enhancements')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(b_fpr, b_tpr, color='orange', label=f'Bug{" "*24}, AUC: {b_roc_auc:.3f}')
        plt.plot(q_fpr, q_tpr, color='blue', label=f'Question{" "*16}, AUC: {q_roc_auc:.3f}')
        plt.plot(e_fpr, e_tpr, color='green', label=f'Enhancement{" "*8}, AUC: {e_roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label=f'Random Guess{" "*6}, AUC: 0.5')
        plt.legend(loc="lower right")
        plt.show()

    b_preds = np.where(b_scores >= b_thres, 1, 0)
    q_preds = np.where(q_scores >= q_thres, 1, 0)
    e_preds = np.where(e_scores >= e_thres, 1, 0)

    b_accuracy = accuracy_score(b_true, b_preds)
    q_accuracy = accuracy_score(q_true, q_preds)
    e_accuracy = accuracy_score(e_true, e_preds)

    b_precision, b_recall, b_f1, _ = precision_recall_fscore_support(b_true, b_preds, average='binary')
    q_precision, q_recall, q_f1, _ = precision_recall_fscore_support(q_true, q_preds, average='binary')
    e_precision, e_recall, e_f1, _ = precision_recall_fscore_support(e_true, e_preds, average='binary')

    y_pred = np.concatenate((b_preds.reshape(-1, 1), q_preds.reshape(-1, 1), e_preds.reshape(-1, 1)), axis=1)

    exact_matches = 0
    for true, pred in zip(y_true, y_pred):
        if (true == pred).all():
            exact_matches += 1

    exact_accuracy = exact_matches/len(y_true)


    metrics_df = pd.DataFrame([ [b_accuracy, b_roc_auc, b_precision, b_recall, b_f1], 
                                [q_accuracy, q_roc_auc, q_precision, q_recall, q_f1], 
                                [e_accuracy, e_roc_auc, e_precision, e_recall, e_f1]], 
                            columns=['Accuracy', 'ROC-AUC', 'Precision', 'Recall', 'F1'], 
                            index=['Bug', 'Question', 'Enhancement'])

    lrap_score = lrap(y_true, model_outputs)


    return metrics_df, exact_accuracy, lrap_score


def load_models_meta():
    with open('models.json') as json_file: 
        models_meta = json.load(json_file)


    return models_meta


def get_n_chunks(df, chunk_size):
    df_size = len(df)

    return int(df_size / chunk_size) if df_size % chunk_size == 0 else int(df_size / chunk_size) + 1


def get_unique_values(df, feature):
    return df.explode(feature)[feature].value_counts()


def df_to_txt(df, return_text=False, save=True, path='./', name='text', extension='txt'):
    text = ''
    for i in range(len(df)):
        title = df.iloc[i].title
        body = df.iloc[i].body

        text += f'{title}\n{body}\n\n'
    
    if save:
        if not path.endswith('/'):
            path = f'{path}/'

        with open(f'{path}{name}.{extension}', 'w') as f:
            f.write(text)
    
    if return_text:
        return text


def split_to_classes(   df, 
                        to_keep=['title', 'body', 'label_bug', 'label_question', 'label_enhancement'], 
                        save=False, 
                        path='./'):

    if to_keep:
        df = df[to_keep]

    bugs_df = df.query('label_bug == 1 and \
                        label_question == 0 and \
                        label_enhancement == 0')

    questions_df = df.query('label_bug == 0 and \
                             label_question == 1 and \
                             label_enhancement == 0')

    enhancements_df = df.query('label_bug == 0 and \
                                label_question == 0 and \
                                label_enhancement == 1')

    combinations_df = df.query('(label_bug == 1 and label_question == 1 and label_enhancement == 0) or \
                                (label_bug == 1 and label_question == 0 and label_enhancement == 1) or \
                                (label_bug == 0 and label_question == 1 and label_enhancement == 1) or \
                                (label_bug == 1 and label_question == 1 and label_enhancement == 1)')

    if save:
        if not path.endswith('/'):
            path = f'{path}/'

        bugs_df.to_pickle(f'{path}bugs.pkl')
        questions_df.to_pickle(f'{path}questions.pkl')
        enhancements_df.to_pickle(f'{path}enhancements.pkl')
        combinations_df.to_pickle(f'{path}combinations.pkl')


    return bugs_df, questions_df, enhancements_df, combinations_df


def get_labels_stats(df):
    total = len(df)

    # all the examples that are labelled
    # only as bugs
    b = len(df.query(   'label_bug == 1 and \
                        label_question == 0 and \
                        label_enhancement == 0'))

    # all the examples that are labelled
    # only as questions
    q = len(df.query(   'label_bug == 0 and \
                        label_question == 1 and \
                        label_enhancement == 0'))

    # all the examples that are labelled
    # only as enhancement
    e = len(df.query(   'label_bug == 0 and \
                        label_question == 0 and \
                        label_enhancement == 1'))

    # all the examples that are labelled
    # only as both bug and question
    b_q = len(df.query('label_bug == 1 and \
                        label_question == 1 and \
                        label_enhancement == 0'))

    # all the examples that are labelled
    # only as both bug and enhancement
    b_e = len(df.query( 'label_bug == 1 and \
                        label_question == 0 and \
                        label_enhancement == 1'))

    # all the examples that are labelled
    # only as both question and enhancement
    q_e = len(df.query( 'label_bug == 0 and \
                        label_question == 1 and \
                        label_enhancement == 1'))

    # all the examples that are labelled
    # as bug, question and enhancement
    b_q_e = len(df.query(   'label_bug == 1 and \
                            label_question == 1 and \
                            label_enhancement == 1'))

    return pd.DataFrame([['Bug', b/total, b], 
                         ['Question', q/total, q], 
                         ['Enhancement', e/total, e], 
                         ['Bug, Question', b_q/total, b_q], 
                         ['Bug, Enhancement', b_e/total, b_e], 
                         ['Question, Enhancement', q_e/total, q_e], 
                         ['Bug, Question, Enhancement', b_q_e/total, b_q_e], 
                         ['Total', total/total, total]], 
                        columns=['Labels Present', 'Fraction', 'Examples'])


def sample_df(  df, 
                n=None, 
                frac=None, 
                to_keep=['title', 'body', 'label_bug', 'label_question', 'label_enhancement']):
    
    assert n != frac

    bugs_df, questions_df, enhancements_df, combinations_df = split_to_classes(df, to_keep)

    if n:
        bugs_df = bugs_df.sample(n=n)
        questions_df = questions_df.sample(n=n)
        enhancements_df = enhancements_df.sample(n=n)
    else:
        bugs_df = bugs_df.sample(frac=frac)
        questions_df = questions_df.sample(frac=frac)
        enhancements_df = enhancements_df.sample(frac=frac)


    # shuffle the dataframes
    bugs_df = bugs_df.sample(frac=1)
    questions_df = questions_df.sample(frac=1)
    enhancements_df = enhancements_df.sample(frac=1)
    combinations_df = combinations_df.sample(frac=1)


    return bugs_df, questions_df, enhancements_df, combinations_df


def split_train_test(   df, 
                        validation=False, 
                        train_frac=.7, 
                        val_frac=.1, 
                        shuffle=True, 
                        save=True, 
                        path='./', 
                        name='', 
                        to_keep=None):
    if shuffle:
        df = df.sample(frac=1)

    if to_keep:
        df = df[to_keep]

    train_split = int(train_frac * (len(df)))

    train_df = df.iloc[:train_split]
    test_df = df.iloc[train_split:]


    if validation:
        val_split = int(val_frac * (len(train_df)))

        val_df = train_df.iloc[:val_split]
        train_df = train_df.iloc[val_split:]


    if save:
        if not path.endswith('/'):
            path = f'{path}/'

        train_df.to_pickle(f'{path}{name}train.pkl')
        test_df.to_pickle(f'{path}{name}test.pkl')

        if validation:
            val_df.to_pickle(f'{path}{name}val.pkl')


    if validation:
        return train_df, val_df, test_df
    else:
        return train_df, test_df


def make_st_compatible(df):
    df['labels'] = list(zip(df.label_bug.tolist(), 
                            df.label_question.tolist(), 
                            df.label_enhancement.tolist()))
    
    for c in df.columns:
        if c.startswith('label_'):
            df = df.drop(c, axis=1)
    
    return df





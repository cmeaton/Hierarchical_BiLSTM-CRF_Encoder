import pandas as pd
import glob
import tensorflow as tf


def load_process_data(directory):
    '''
    This function inputs 'directory', which is the directory where you saved all data from 'swda_parsed/'
    '''

    file_list = []
    for f in glob.glob(directory):
    file_list.append(f)

    df_list = []
    for i in file_list:
    df = pd.read_csv(i)
    df_list.append(df)

    text_list = []
    label_list = []

    for df in df_list:
    df['utterance_no_specialchar_'] = df.utterance_no_specialchar.astype(str)
    text = df.utterance_no_specialchar_.tolist()
    labels = df.da_category.tolist()
    text_list.append(text)
    label_list.append(labels)

    return text_list, label_list

def custom_tokenize(text_list):
    '''The hierarchical nature of this model requires the specific text structure preserved. 
    This code allows for corpus-wide tokenization of text while preserving structure.
    '''

    text_list = [[[j] for j in i] for i in text_list]

    tok_data = [y[0] for x in text_list for y in x]

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(tok_data)

    sequences = []
    for x in text_list:
    tmp = []
    for y in x:
        tmp.append(tokenizer.texts_to_sequences(y)[0])
    sequences.append(tmp)

    return sequences

import re
import string
import pandas as pd
import numpy as np
import tensorflow as tf

def pd_text_standardization(text):
    """
    Função usada para padronizar uma entrada de textos no formato de dataframe.

    Args:
        text (dataframe): dataframe de textos.

    Returns:
        text (dataframe): texto padronizado.
    """
    text = text.str.lower()
    text = text.str.replace('-',' ')
    for i in range(text.shape[0]):        
        text.loc[i] = "".join(c for c in text.loc[i] if c.isalpha() or c==' ')
    others = ['ª','  ']
    for c in others:
        text = text.str.replace(c,'')        
    return text

def tf_text_standardization(text):
    """
    Função usada para padronizar uma entrada de textos no formato de tensor.

    Args:
        text (tensor): um tensor contendo textos.

    Returns:
        text (tensor): texto padronizado.
    """    
    #text = tf.squeeze(text)
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text,'-',' ')
    punctuations = re.escape(string.punctuation)
    text = tf.strings.regex_replace(text,f'[{punctuations}]','')
    return text
    #return tf.expand_dims(text,-1)
    
        
def df_text_statistics(text):
    """
    Função usada para gerar estatísticas sobre uma dataframe de textos.

    Args:
        text (dataframe): dataframe de textos.

    Returns:
        (dict): um dicionário contento estatísticas sobre o texto dado.
    """    
    word_count = text.str.split(' ').apply(len)
    max_word_count = int(word_count.max())
    median_word_count = int(word_count.median())
    total_words = word_count.sum()
    return {'word_count':word_count,'total_words':total_words,
        'median_word_count':median_word_count,'max_word_count':max_word_count}
    
def tf_text_statistics(text):
    """
    Função usada para gerar estatísticas sobre uma tensor de textos.

    Args:
        text (tensor): tensor de textos.

    Returns:
        (dict): um dicionário contento estatísticas sobre o texto dado.
    """    
    text = tf.squeeze(text)
    word_count = tf.strings.split(text,' ')
    word_count = tf.map_fn(lambda x: x.shape[0],word_count,fn_output_signature=tf.int32)
    max_word_count = np.max(word_count)
    median_word_count = np.median(word_count)
    total_words = np.sum(word_count)    
    # Conta vocabulário
    words = tf.strings.join(text,' ')
    words = tf.strings.split(words,' ')
    vocab,_,vocab_freq = tf.unique_with_counts(words)
    vocab_size = vocab.shape[0]
    
    return {'vocab':vocab,'vocab_freq':vocab_freq,'vocab_size':vocab_size,
        'word_count':word_count,'total_words':total_words,
        'median_word_count':median_word_count,'max_word_count':max_word_count}
      
def unique_classes(labels):
    """
    Função usada para extrair rótulos exclusivos de uma dataframe de rótulos.

    Args:
        labels (dataframe): dataframe de rótulos.

    Returns:
        classes (dataframe): dataframe contendo rótulos exclusivos e seu identificador.
    """
    classes = list(map(lambda x: x.split(','),labels.unique()))
    classes = " - ".join(labels.unique())
    classes = classes.replace(',',' - ')
    classes = pd.unique(classes.split(' - '))
    classes = pd.DataFrame([range(len(classes))],columns=sorted(classes))
    size = classes.shape[1]
    #print('Number of classes: ',size)    
    return classes

def encode_labels(labels,classes):
    """
    Transforma um dataframe de rótulos em um array de 0's e 1's.

    Args:
        labels (dataframe): dataframe de rótulos.
        classes (dataframe): dataframe contendo os rótulos exclusivos com identificador.

    Returns:
        encoded_labels (array): array contendo os rótulos codificados.
    """
    encoded_labels = np.zeros((labels.shape[0],classes.shape[1]))
    for i in range(encoded_labels.shape[0]):
        l = labels.loc[i].split(',')
        encoded_labels[i,classes[l]] = 1 
    return encoded_labels

def decode_strings(text):
    """
    Converte uma lista de byte strings para uma lista de strings.

    Args:
        text (list): lista de textos no formato de byte string.

    Returns:
        (list): lista de textos no formato de string.
    """
    return [x.decode() for x in text.numpy()]
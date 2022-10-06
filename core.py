import os
import numpy as np
import pandas as pd
import tensorflow as tf

import util

class TextClassification:
    """
    Classe desenvolvidade para resolver o problema de classificação de texto multi-rótulo.
    
    Versão: 1.0 (Outubro 2022)
    Autor: Samuel Silva (samuelrs@usp.br)
    """    
    def __init__(self,dataset=None,text_col=None,labels_col=None,select_vocab=True):
        """
        Construtor.

        Args:
            dataset (dataframe): dataset no formato de dataframe pandas.
            text_col (str): coluna do dataframe correspondente a lista de textos.
            labels_col (str): coluna do dataframe correspondente a lista de labels.
            select_vocab (bool): True para criar vocabulário customizado ou False 
                para criar vocabulário completo.
        """
        if dataset is None:
           self.load_vectorization_configs()
        else:
            text,labels = dataset[text_col],dataset[labels_col]
            
            # Padronização do texto
            self.standardized_text = util.tf_text_standardization(text)
            
            # Gera estatísticas sobre o texto padronizado
            self.text_statistics = util.tf_text_statistics(self.standardized_text)
            self.classes = util.unique_classes(labels)
            
            self.encoded_labels = util.encode_labels(labels,self.classes)
            
            self.vocab_freq = pd.DataFrame(self.text_statistics['vocab_freq'],index=self.text_statistics['vocab'],columns=['freq'])
            #self.vocab_freq.to_csv('vocab_freq')
            
            if select_vocab:
                self.selected_vocab = self.vocab_freq.index[np.logical_and(0<self.vocab_freq.freq,self.vocab_freq.freq<20)]
                self.selected_vocab = tf.convert_to_tensor(self.selected_vocab.to_list(),dtype=tf.string)
                #self.selected_vocab = self.vocab_freq.index[:int(0.98*self.vocab_freq.shape[0])].to_list()         
                self.max_tokens = len(self.selected_vocab) + 2                
            else:
                self.selected_vocab = None
                self.max_tokens = self.text_statistics['vocab_size']+1
            
            self.output_sequence_length = int(self.text_statistics['median_word_count'])
            #self.embedding_dim = int(0.5*self.output_sequence_length)
            self.embedding_dim = 5 
            
            self.save_vectorization_configs()
                    
        # Camada de vetorização de textos
        self.vect_layer =  tf.keras.layers.TextVectorization(
            #standardize = "lower_and_strip_punctuation",
            standardize = util.tf_text_standardization,
            #ngrams = 5,
            max_tokens = self.max_tokens,
            vocabulary = self.selected_vocab,
            output_mode = "int",
            output_sequence_length = self.output_sequence_length)
        
        # Inicializa tabela de vocabulário caso o vocabulário não tenha sido especificado
        if not select_vocab:
            self.vect_layer.adapt(self.standardized_text)
        
    def create_train_validation_datasets(self,batch_size=16,train_size=0.8):
        """
        Método usado para criar os datasets de treino e validação usando a api tf.data.Dataset.

        Args:
            batch_size (int): tamanho do batch.
            train_size (float): proporção de amostras usadas para treino.            
        """
        
        dataset = tf.data.Dataset.from_tensor_slices((self.standardized_text,self.encoded_labels))        
        dataset = dataset.map(lambda text,labels: (tf.expand_dims(text,-1),labels))
        #dataset = dataset.map(lambda text,labels: (self.vect_layer(text),labels))
        
        trainset,valset = tf.keras.utils.split_dataset(dataset,train_size,1-train_size,True)
        
        trainset = trainset.batch(batch_size)
        trainset = trainset.shuffle(10*batch_size, reshuffle_each_iteration=True)
        self.trainset = trainset.repeat(5)
        
        self.valset = valset.batch(batch_size)
    
    def build_model(self):
        """
        Método usado pra criar a arquitetura da Rede Neural.
        OBS: Não testei muitas variações.
        """
        inputs = tf.keras.Input(shape=(1,), dtype=tf.string)

        x = self.vect_layer(inputs)

        x = tf.keras.layers.Embedding(self.max_tokens,self.embedding_dim)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        x = tf.keras.layers.Conv1D(16, 3, padding="same" , strides=1, activation="relu")(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(8, 3, padding="same" , strides=1, activation="relu")(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)                                                                        
        x = tf.keras.layers.Dropout(0.5)(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(10,activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        outputs = tf.keras.layers.Dense(self.classes.shape[1], activation="sigmoid")(x)

        self.model = tf.keras.Model(inputs, outputs) 

        self.model.summary()
                                
    def train(self,epochs=10,learning_rate=0.01,device='cpu'):
        """
        Método usado para iniciar o treinamento da Rede Neural.

        Args:
            epochs (int): número de épocas de treino.
            learning_rate (float): taxa de aprendizagem.
            device (str): hardware usado para treinar a rede.
        """                
        self.model.compile(
            loss = "binary_crossentropy",
            optimizer = tf.keras.optimizers.Adam(learning_rate),
            metrics = ["accuracy"])   
        
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=20)
        
        with tf.device(device):
            self.model.fit(
                self.trainset,
                validation_data = self.valset,
                epochs = epochs,
                callbacks = [early_stopping_callback])
                        
        model_file = os.path.join(os.path.dirname(__file__),"model_weights.h5")
        self.model.save_weights(model_file)

    def predict(self,text,device='cpu'):
        """
        Método usado para predizer as categorias de textos.

        Args:
            text (list): lista de textos a serem classificados.
            device (str): hardware usado para realizar a predição.

        Returns:
            (dataframe): dataframe pandas com as previsões.
        """
        with tf.device(device):
            prediction = self.model.predict(text)                        
        return pd.DataFrame(prediction,columns=self.classes.columns)

    def load_weights(self,weights_file):
        """
        Método usado para carregar pesos pré-treinados.

        Args:
            weights_file (str): nome do arquivo de peos. 
        """
        self.model.load_weights(weights_file)        
        
    def save_vectorization_configs(self):
        """
        Método usado para salvar as configurações da camada de vetorização de textos.
        """        
        vect_config_path = 'vectorization_configs'
        os.makedirs(vect_config_path,exist_ok=True) 
        
        # Salva classes
        self.classes.to_csv(os.path.join(vect_config_path,'classes.csv'),index=False)        
        
        # Salva vocabulário selecionado
        sel_vocab = pd.DataFrame(self.selected_vocab,columns=['vocabulary'])
        sel_vocab.to_csv(os.path.join(vect_config_path,'selected_vocabulary.csv'),index=False)
        
        # Salva configuraçoes da camada de vetorização de texto
        vect_configs = pd.DataFrame(
            [[self.max_tokens,self.output_sequence_length,self.embedding_dim]],
            columns=['max_tokens','output_sequence_length','embedding_dim'])
        vect_configs.to_csv(os.path.join(vect_config_path,'vectorization_configs.csv'),index=False)    
        
    def load_vectorization_configs(self):
        """
        Método usado para carregar as configurações da camada de vetorização de textos.
        """             
        vect_config_path = 'vectorization_configs'
        
        # Carrega classes
        self.classes = pd.read_csv(os.path.join(vect_config_path,'classes.csv'))
        
        # Carrega vocabulário
        self.selected_vocab = pd.read_csv(os.path.join(vect_config_path,'selected_vocabulary.csv'))
        self.selected_vocab = tf.convert_to_tensor(self.selected_vocab['vocabulary'].values,dtype=tf.string)
        
        # Carrega configurações da camada de vetorização de texto
        vect_config = pd.read_csv(os.path.join(vect_config_path,'vectorization_configs.csv'))
        self.max_tokens = int(vect_config.loc[0,'max_tokens'])
        self.embedding_dim = int(vect_config.loc[0,'embedding_dim'])
        self.output_sequence_length = int(vect_config.loc[0,'output_sequence_length'])
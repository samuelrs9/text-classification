import sys
import pandas as pd
from neural_net import TextClassification
    
def train(dataset_file,text_col,labels_col):
    """
    Função usada para configurar a inicialização do classificador de textos
    e treiná-lo.

    Args:
        dataset (dataframe): dataset no formato de dataframe pandas.
        text_col (str): coluna do dataframe correspondente a lista de textos.
        labels_col (str): coluna do dataframe correspondente a lista de labels.
    """    
    dataset = pd.read_csv(dataset_file)
    tc = TextClassification(dataset,text_col,labels_col)
    tc.create_train_validation_datasets(batch_size=8,train_size=0.9)
    tc.build_model()
    tc.train(epochs=20,learning_rate=0.002)

if __name__ == "__main__":
    if len(sys.argv)==1:
        train("dataset.csv","sentence","category")
    elif len(sys.argv)==4:
        train(sys.argv[1],sys.argv[2],sys.argv[3])
    else:
        raise Exception("\n>> Uso incorreto de argumentos.\n>> Tente: "
            "python train.py <dataset_name.csv> <text_column> <labels_column>")
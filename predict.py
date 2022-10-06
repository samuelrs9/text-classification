import sys
import pandas as pd
from neural_net import TextClassification
    
def predict():
    """
    Função usada para configurar a inicialização do classificador de textos
    e realizar predições. É preciso que o classificador tenha sido treinado
    previamente.
    """
    tc = TextClassification()    
    tc.build_model()
    tc.load_weights('model_weights.h5')
        
    while True:
        text = input("\nDIGITE UMA FRASE E TECLE ENTER: ")
        prediction = tc.predict([text])
        print('\nProbabilidade da frase se referir as categorias abaixo:')
        print(prediction.head())

if __name__ == "__main__":
    predict()
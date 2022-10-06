# Text Classification
Esse repositório contém uma implementação de Classificação de Textos Multi-Rótulo usando Rede Neural Convolucional Unidimensional (CNN 1D).
## Preparação do ambiente
Esse projeto foi testado utilizando Python 3.10 no Windows 10 com o seguinte ambiente Miniconda3
```
conda create -n nlp-env python=3 pip tensorflow pandas matplotlib
conda activate nlp-env
```
## Dataset
Na raíz do repositório pode ser encontrado um arquivo chamado "dataset.csv". Esse dataset contém frases rotuladas com as seguintes categorias: **educação**, **finanças**, **indústrias**, **orgão público** e **varejo**.

Essa base possui apenas 521 amostras de frases, o que pode dificultar bastante a generalização de modelos de classificação. 

## Treino
Para treinar o modelo com a base "dataset.csv" basta executar o seguinte comando
``` 
train.py dataset.csv sentence category
```

Com a arquitetura testada até o momento o treinamento converge bem no conjunto de treino, porém o desempenho não é replicado para conjunto de validação. Isso provavelmente está ocorrendo  devido a baixa quantidade de amostras no dataset.

## Predição
A classificação de textos pode ser feita de forma interativa passando frases através do terminal. Para tanto, execute o seguinte script
```
predict.py
``` 

## TODO
* Testar mais variações de hiperparâmetros na arquitetura da rede neural.
* Testar algum esquema de data augmentation.
* Verificar se a camada de vetorização de texto está funcionando adequadamente.
* Pensar em formas de escolher um vocabulário mais descritivo.
* Testar com outros datasets.

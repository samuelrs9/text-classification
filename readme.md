# Text Classification
Esse repositório contém uma implementação de Classificação de Textos Multi-Rótulo usando Rede Neural Convolucional Unidimensional (CNN 1D).
## Preparação do ambiente
Esse projeto foi testado utilizando Python 3.10 no Windows 10 com o seguinte ambiente Miniconda3
```
conda create -n nlp-env python=3 pip tensorflow pandas matplotlib
conda activate nlp-env
```
## Dataset
Na raíz do repositório pode ser encontrado um arquivo chamado dataset.csv. Esse dataset contém frases rotuladas com as seguintes categorias: **educação**, **finanças**, **indústrias**, **orgão público** e **varejo**.

Essa base possui apenas 521 amostras de frases, o que pode dificultar bastante a generalização do modelo. 

## Treino
Para treinar o modelo na base "dataset.csv" basta executar o seguinte comando
``` 
train.py dataset.csv sentence category
```

## Predição
A classificação de textos pode ser feita de forma interativa passando frases através do termnal. Execute
```
predict.py
``` 

## TODO
* Testar mais configurações de arquiteturas.
* Testar com outros datasets.

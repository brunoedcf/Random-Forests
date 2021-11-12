from os import terminal_size
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from math import sqrt
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score, classification_report

print("\n\n\n")

data = pd.read_csv("diabetes.csv") #data = dataframe from pandas

X = data.drop("Outcome", axis=1) #axis = 1 indica colunas do dataframe
y = data["Outcome"]

# x e y sao dataframes do pandas
# x sao as variaveis = sintomas ou dados
# y sao os resultados = diagnostico

estimadores = [10, 100, int(sqrt(768))] # n = 10, 100 e sqrt do estudo do algoritmo

for est in estimadores:
    #loop para gerar os resultados de todos os estimadores descritos acima
    print(f'RESULTADO PARA {est} ESTIMADORES')

    #separar os 80% para treino e 20% para teste
    Xtraining, Xtest, ytraining, ytest = train_test_split(X, y, test_size=0.2, random_state=0) #test_size = 0.2 seleciona 20% para teste e 80% para treino

    #geramos e treinamos a floresta com os dados acima
    floresta = RandomForestClassifier(n_estimators=est, random_state=0, max_features="sqrt") 
    floresta.fit(Xtraining, ytraining)

    #comparamos a predição da floresta treinada com os resultados verdadeiros para medir a precisão
    pred = floresta.predict(Xtest)
    precisao = accuracy_score(ytest, pred)

    print(precisao)
    #aqui são mostradas as medidas de revocação e medida f1
    print(classification_report(ytest, pred))

    #separando do dataframe as colunas das variaveis de importancia para usar de parametro para a função de importancia
    importancia = pd.DataFrame({'feature': list(Xtraining.columns), 'importance': floresta.feature_importances_}).\
    sort_values('importance', ascending = False)
    #ordenar de forma decrescente

    #utilizar o parametro 'importance' para separar as variaveis por importancia
    #ascending = false em sort_values() para manter as variaveis de maior valor de importancia em primeiro

    print("Variaveis independentes ordenadas pelo valor de sua importância:")
    print(importancia)

    #Gerar matriz e confusao
    matrizconfusao = ConfusionMatrix(floresta)
    matrizconfusao.fit(Xtraining, ytraining)
    matrizconfusao.score(Xtest, ytest)
    #mostrar cada matriz de confusão gerada por cada um dos estimadores
    matrizconfusao.show()
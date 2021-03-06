import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors

simplefilter(action='ignore', category=FutureWarning)

url ='weatherAUS.csv'
data = pd.read_csv(url)

data.drop(['Date','MinTemp','MaxTemp','Rainfall','Evaporation',
           'Pressure9am','Sunshine','Pressure3pm','Temp9am','Temp3pm',
           'RISK_MM'],axis=1, inplace=True)

data['Location'].replace(['Canberra','Sydney','Perth','Darwin',
'Hobart','Brisbane','Adelaide','Bendigo','Townsville',
'AliceSprings','MountGambier','Launceston','Ballarat','Albany',
'Albury','PerthAirport','MelbourneAirport','Mildura','SydneyAirport',
'Nuriootpa','Sale','Watsonia','Tuggeranong','Portland','Woomera','Cairns',
'Cobar','Wollongong','GoldCoast','WaggaWagga','Penrith','NorfolkIsland',
'SalmonGums','Newcastle','CoffsHarbour','Witchcliffe','Richmond',
'Dartmoor','NorahHead','BadgerysCreek','MountGinini','Moree',
'Walpole','PearceRAAF','Williamtown','Melbourne','Nhil',
'Katherine','Uluru'], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,
                       18,19,20,21,22,23,24,25,26,27,28,29,30,
                       31,32,33,34,35,36,37,38,39,40,
                       41,42,43,44,45,46,47,48], inplace=True)

                       
data['WindGustDir'].replace(['W','SE','E','N','SSE','S','WSW','SW',
                     'SSW','WNW','NW','ENE','ESE','NE','NNW',
                     'NNE'], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], inplace=True)

data['WindDir9am'].replace(['W','SE','E','N','SSE','S','WSW','SW',
                     'SSW','WNW','NW','ENE','ESE','NE','NNW',
                     'NNE'], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], inplace=True)

data['WindDir3pm'].replace(['W','SE','E','N','SSE','S','WSW','SW',
                     'SSW','WNW','NW','ENE','ESE','NE','NNW',
                     'NNE'], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], inplace=True)

data.RainToday.replace(['Yes','No'], [1,0], inplace=True)

data.RainTomorrow.replace(['Yes','No'], [1,0], inplace=True)


data.dropna(axis=0,how='any',inplace=True) 

#partir la tabla en dos
data_train = data[:80001]
data_test = data[80001:]

x=np.array(data_train.drop(['RainTomorrow'], 1))
y=np.array(data_train.RainTomorrow)# 0 no llueve 1 llueve

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.5)

x_test_out = np.array(data_test.drop(['RainTomorrow'], 1))
y_test_out = np.array(data_test.RainTomorrow)# 0 no llueve 1 llueve


# seleccionar modelo regresion logistica
logreg = LogisticRegression(solver='lbfgs', max_iter=7600)

#entreno el modelo
logreg.fit(x_train, y_train)

#metricas

print('*'*50)
print('Regresion Logistica')

#accuracy de entrenamiento train
print(f'accuracy de train de entrenamiento:{logreg.score(x_train, y_train)}')
#accuracy de entrenamiento test
print(f'accuracy de test de entrenamiento:{logreg.score(x_test,y_test)}')
#accuracy de validacion
#print(f'accuracy de validacion:{logreg.score(x_test_out,y_test_out)}')

# seleccionar maquina soporte vectorial
svc = SVC(gamma='auto')

#entreno el modelo
svc.fit(x_train, y_train)

#metricas

print('*'*50)
print('Maquina de soporte vectorial')

#accuracy de entrenamiento train
print(f'accuracy de train de entrenamiento:{svc.score(x_train, y_train)}')
#accuracy de entrenamiento test
print(f'accuracy de test de entrenamiento:{svc.score(x_test,y_test)}')
#accuracy de validacion
#print(f'accuracy de validacion:{svc.score(x_test_out,y_test_out)}')



# seleccionar modelo arbol de decisiones
arbol = DecisionTreeClassifier()

#entreno el modelo
arbol.fit(x_train, y_train)

#metricas

print('*'*50)
print('arbol de decisiones|')

#accuracy de entrenamiento train
print(f'accuracy de train de entrenamiento:{arbol.score(x_train, y_train)}')
#accuracy de entrenamiento test
print(f'accuracy de test de entrenamiento:{arbol.score(x_test,y_test)}')
#accuracy de validacion
#print(f'accuracy de validacion:{arbol.score(x_test_out,y_test_out)}')


# seleccionar modelo Naive Bayes
gnb = GaussianNB()

#entreno el modelo
gnb.fit(x_train, y_train)

#metricas

print('*'*50)
print('Naive Bayes')

#accuracy de entrenamiento train
print(f'accuracy de train de entrenamiento:{gnb.score(x_train, y_train)}')
#accuracy de entrenamiento test
print(f'accuracy de test de entrenamiento:{gnb.score(x_test,y_test)}')
#accuracy de validacion
#print(f'accuracy de validacion:{gnb.score(x_test_out,y_test_out)}')

# seleccionar modelo KNN
knn = neighbors.KNeighborsClassifier()

#entreno el modelo
knn.fit(x_train, y_train)

#metricas

print('*'*50)
print('modelo KNN')

#accuracy de entrenamiento train
print(f'accuracy de train de entrenamiento:{knn.score(x_train, y_train)}')
#accuracy de entrenamiento test
print(f'accuracy de test de entrenamiento:{knn.score(x_test,y_test)}')
#accuracy de validacion
#print(f'accuracy de validacion:{knn.score(x_test_out,y_test_out)}')

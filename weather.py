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

# -*- coding: utf-8 -*-
"""

@author: ahmed
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:12:03 2019

@author: ahmed
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#################################################################
# Loading my data
file_data = 'training.csv'
dataset = pd.read_csv(file_data,sep = ';',decimal=',')


X_train = pd.DataFrame(dataset.iloc[:, :-1].values)
y_train = pd.DataFrame(dataset.iloc[:, -1].values)



file_data = 'validation.csv'
dataset = pd.read_csv(file_data,sep = ';',decimal=',')




X_test = pd.DataFrame(dataset.iloc[:, :-1].values)
y_test = pd.DataFrame(dataset.iloc[:, -1].values)

del file_data
###################################################################
# Identify my columns datatypes to deal with NaN later
string_colums = []
decimal_colums = []
integar_colums = []

for i in range(X_train.shape[1]):
    mType = type(X_train[i][0])
    if mType == str:
        string_colums.append(i)
    elif mType == int:
        integar_colums.append(i)
    else:
        decimal_colums.append(i)

        
number_colums = list(np.concatenate((integar_colums,decimal_colums)))
        
#######################################################################
# Filling NaNs with reasonable values      
# use most frequent with str
# use mean with floats
# use median with integars        
Imputer_X = SimpleImputer(missing_values= np.nan,strategy="median")
X_train = X_train.values
X_test = X_test.values
X_train[:,integar_colums] = Imputer_X.fit_transform(X_train[:,integar_colums])
X_test[:,integar_colums] = Imputer_X.transform(X_test[:,integar_colums])

Imputer_X = SimpleImputer(missing_values= np.nan ,strategy="mean")
X_train[:,decimal_colums] = Imputer_X.fit_transform(X_train[:,decimal_colums])
X_test[:,decimal_colums] = Imputer_X.transform(X_test[:,decimal_colums])


X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)



most_freq_train = X_train[string_colums].mode()
most_freq_test = X_test[string_colums].mode()

for i in string_colums:
    X_train[i] = X_train[i].fillna(value = most_freq_train[i][0])
    X_test[i] = X_test[i].fillna(value = most_freq_test[i][0])
   


del most_freq_train,most_freq_test



##############################################################
# Encoding both my Targets and some features


labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(np.ravel(y_train))
y_test = labelencoder_y.transform(np.ravel(y_test))



temp_list = []
for i in string_colums:
    temp_list.append(X_train[i].unique())

for i in string_colums:
    temp_list.append(X_test[i].unique())




temp2 = []


for i in range(len(temp_list)):
    for j in range(len(temp_list[i])):
        temp2.append(temp_list[i][j])
        
del temp_list       

# set doesn't store the same element twice
temp2 = set(temp2) 
 # convert the set to the list 
temp2 = (list(temp2)) 


        
labelencoder_X = LabelEncoder()
labelencoder_X.fit(temp2)
#labelencoder_X.transform(temp2)
del temp2

for i in string_colums:
    X_train[i] = labelencoder_X.transform(X_train[i])
    X_test[i] = labelencoder_X.transform(X_test[i])
 
del i,j


###############################################################
# Feature selection section
test = SelectKBest(score_func=chi2, k=7)
fit = test.fit(X_train, y_train)
features_tr = fit.transform(X_train)
features_ts = fit.transform(X_test)

#############################################################
# Scaling my Input 
scaler = StandardScaler()
features_tr  =    scaler.fit_transform(features_tr) 
features_ts  =    scaler.transform(features_ts) 

#features_tr = features_tr[:600,:]
#y_train = y_train[:600]

##############################################################
# Running my model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators =8, criterion = 'entropy', random_state = 0)
classifier.fit(features_tr, y_train)


y_pred = classifier.predict(features_ts)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)

y_pred = classifier.predict(features_tr)

cm2 = confusion_matrix(y_train, y_pred)



acc1 = (cm1[0][0] + cm1[1][1]) / np.sum(np.sum(cm1))
acc2 = (cm2[0][0] + cm2[1][1]) / np.sum(np.sum(cm2))

prec1 = (cm1[0][0]) /(cm1[0][0] + cm1[0][1])
prec2 = (cm2[0][0]) /(cm2[0][0] + cm2[0][1])

Rec1 = (cm1[0][0]) /(cm1[0][0] + cm1[1][0])
Rec2 = (cm2[0][0]) /(cm2[0][0] + cm2[1][0])

spec1 = (cm1[1][1]) /(cm1[0][1] + cm1[1][1])
spec2 = (cm2[1][1]) /(cm2[0][1] + cm2[1][1])

performance = {
        'accuracy' : [acc1,acc2],
        'Precision': [prec1,prec2],
        'Recall' : [Rec1,Rec2],
        'Specificity' : [spec1,spec2]
        
        }

performance = pd.DataFrame(performance, columns=['accuracy','Precision','Recall','Specificity'])

del acc1,acc2,prec1,prec2,Rec1,Rec2,spec1,spec2

# End here
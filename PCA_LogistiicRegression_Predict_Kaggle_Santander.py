# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:17:43 2022

@author: Devashish
"""

import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir(r"D:\CDAC ML\Cases\Kaggle\Santander Customer Satisfaction")


train = pd.read_csv(r'train.csv',index_col =0)

test = pd.read_csv(r'test.csv',index_col =0)

X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.15,
                                               random_state=2022,
                                               stratify=y)

scaler = StandardScaler()
milkscaled = scaler.fit_transform(train)
milk.shape

pca = PCA()
principalcomponents = pca.fit_transform(milkscaled)
principalcomponents
principalcomponents.shape

print(pca.explained_variance_)
print(pca.explained_variance_ratio_*100)

print(np.cumsum(pca.explained_variance_ratio_)*100)


#### scree plot
import matplotlib.pyplot as plt
ys=np.cumsum((pca.explained_variance_ratio_)*100)
xs=np.arange(1,26)
plt.plot(xs,ys)
plt.show()

#### seleccting 1st 10 PCA
from sklearn.linear_model import LogisticRegression

PCX = pd.DataFrame(principalcomponents[:,:10])

model=LogisticRegression()
model.fit(PCX,y_train)

### 


#### processing test

X_test_scaled = scaler.transform(X_test)

pc_X_test=pca.transform(X_test_scaled)[:,:10]
y_pred = model.predict(pc_X_test)


#### model evaluation
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

#### ROC
from sklearn.metrics import roc_curve,roc_auc_score

## compute predicted probalities y pred prob

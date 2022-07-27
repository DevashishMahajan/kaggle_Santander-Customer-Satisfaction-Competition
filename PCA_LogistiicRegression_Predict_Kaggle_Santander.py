# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:17:43 2022

@author: Devashish
"""

#Importing necessary libraries
import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Change the file path as the file path in your computer
import os
os.chdir(r"D:\CDAC ML\Cases\Kaggle\Santander Customer Satisfaction")

# Read csv files from kaggle dataset as Pandas Dataframe
train = pd.read_csv(r'train.csv',index_col =0)

test = pd.read_csv(r'test.csv',index_col =0)

# X is a feature
X_train = train.iloc[:,:-1]

# y is a label
y_train = train.iloc[:,-1]

# Train test split
X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.15,
                                               random_state=2022,
                                               stratify=y)

# Standard Scaler
scaler = StandardScaler()
milkscaled = scaler.fit_transform(train)
milk.shape

# Principal component analysis
pca = PCA()
principalcomponents = pca.fit_transform(milkscaled)
principalcomponents
principalcomponents.shape

print(pca.explained_variance_)
print(pca.explained_variance_ratio_*100)

print(np.cumsum(pca.explained_variance_ratio_)*100)


#### Scree plot
import matplotlib.pyplot as plt
ys=np.cumsum((pca.explained_variance_ratio_)*100)
xs=np.arange(1,26)
plt.plot(xs,ys)
plt.show()

#### Selecting 1st 10 PCA
from sklearn.linear_model import LogisticRegression

# Convert to pandas dataframe
PCX = pd.DataFrame(principalcomponents[:,:10])

# Logistic Regression
model=LogisticRegression()
model.fit(PCX,y_train)

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




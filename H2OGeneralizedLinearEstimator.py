# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:17:13 2022

@author: Devashish
"""
import pandas as pd
import h2o
h2o.init()

#kaggle API to download dataset 
#kaggle competitions download -c santander-customer-satisfaction

df=h2o.import_file(r"D:\CDAC ML\Cases\Kaggle\Santander Customer Satisfaction\train.csv",destination_frame="Santander_train")

test=h2o.import_file(r"D:\CDAC ML\Cases\Kaggle\Santander Customer Satisfaction\test.csv",destination_frame="Santander_test")

print(df.col_names)

y = 'TARGET'
X = df.col_names[1:-1]

X_test=test.col_names[1:]

print('Response = ' +y)
print('Pridictors = '+ str(X))

#because D is categotical
df['TARGET'] = df['TARGET'].asfactor()
df['TARGET'].levels()

#train, test= df.split_frame(ratios=[0.7],seed=2022)
print(df.shape)
print(train.shape)
print(test.shape)

### logistic
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
glm_logistic = H2OGeneralizedLinearEstimator(family="binomial")

glm_logistic.train(x=X, y=y,training_frame=df,
                  model_id="glm_logistic")

y_pred = glm_logistic.predict(test_data=test)

y_pred_df= y_pred.as_data_frame() #convert into pandas dataframe

print(glm_logistic.auc())
print(glm_logistic.confusion_matrix())
#h2o.cluster().shutdown() #close resources

submit=pd.concat([test['ID'].as_data_frame(),y_pred_df[['p1']]],axis=1)
submit.rename(columns={'p1':'TARGET'},inplace=True)
submit.to_csv('Santander_H2O_logistic.csv',index=False)

###naive 
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator

nb= H2ONaiveBayesEstimator()

nb.train(x=X, y=y,training_frame=df,
                   validation_frame=test,model_id="Naive_bayes")

y_pred = nb.predict(test_data=test)

y_pred_df= y_pred.as_data_frame() #convert into pandas dataframe

print(nb.auc())
print(nb.confusion_matrix())
#h2o.cluster().shutdown() #close resources



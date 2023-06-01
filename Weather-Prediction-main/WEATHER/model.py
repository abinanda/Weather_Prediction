#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('weather.csv')
df = df.drop('date',axis=1)
X = df.drop('weather',axis=1)
y = df['weather']


X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size =0.25, random_state =42)


gbc=GradientBoostingClassifier(subsample=0.5,n_estimators=200,max_depth=10,max_leaf_nodes=50)
gbc.fit(X_train,y_train)
gbc.predict(X_test)




pickle.dump((gbc), open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


# In[ ]:





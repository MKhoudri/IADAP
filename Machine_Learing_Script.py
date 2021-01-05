#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from neuraxle.steps.loop import FlattenForEach
from neuraxle.steps.column_transformer import ColumnTransformer


# In[2]:


# lecture du corpus
train = pd.read_csv('CorpusARD.csv')
#train = train.loc[train['#PAYS']=='Djibouti']
train['target'] = train['#PAYS']
train.head(7)


# In[3]:


# la transformation et la répartition du corpus en corpus d'aprentissage 70% et corpus test 30% 
list_var_cat = train.columns.tolist()       
enc = LabelEncoder()

for i in list_var_cat:
    train[i] = enc.fit_transform(train[i])

print("reminder taille de train avant: ", train.shape)
train, test = train_test_split(train,test_size=0.3,shuffle = False)
print("taille de train: ", train.shape, "taille de test: ", test.shape)


# In[4]:


# le retrait de la cible dans le corpus (apprentissage/test)
y_train = train[['target']].copy()
x_train = train[['#ID_ENONCE', '#ID_TOKEN', '#LEN_TOKEN', '#TOKEN', '#PAYS', '#PROVINCE','#ENONCE']].copy()

y_test = test[['target']].copy()
x_test = test[['#ID_ENONCE', '#ID_TOKEN', '#LEN_TOKEN', '#TOKEN', '#PAYS', '#PROVINCE','#ENONCE']].copy()
y_train = train[['target']].copy()
x_train = train[['#ID_ENONCE', '#ID_TOKEN', '#LEN_TOKEN', '#TOKEN', '#PAYS', '#PROVINCE','#ENONCE']].copy()

y_test = test[['target']].copy()
x_test = test[['#ID_ENONCE', '#ID_TOKEN', '#LEN_TOKEN', '#TOKEN', '#PAYS', '#PROVINCE','#ENONCE']].copy()


# In[5]:


# le choix du corpus d'apprentissage en choisissant les colonnes à abondonner
x_train.drop(['#PAYS','#PROVINCE','#ID_ENONCE', '#ID_TOKEN', '#LEN_TOKEN', '#TOKEN'], axis=1, inplace=True)
x_test.drop(['#PAYS','#PROVINCE','#ID_ENONCE', '#ID_TOKEN', '#LEN_TOKEN', '#TOKEN'], axis=1, inplace=True)
x_train.columns


# In[6]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=4)
#KEEP les variables à utiliser et .fit sur la target
rf.fit(x_train, y_train)
pred = rf.predict(x_test)


# In[7]:


#évaluation du modèle de prédiction
from sklearn.metrics import accuracy_score, f1_score


acc = accuracy_score(y_test, pred)
fscore = f1_score(y_test, pred, average="micro")

print("accuracy: ", acc)
print("f_score: ", fscore)


# In[ ]:





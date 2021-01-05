#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import pyarabic.araby as araby
from lang_trans.arabic import buckwalter
import re
import string


# In[6]:


# lecture des données initial
corpus = pd.read_csv("train_labeled.tsv",sep='\t',encoding = 'utf8')
corpus.head(20)


# In[7]:


# récuperation des données dans des liste
list_enonce = list(corpus["#2 tweet_content"])
list_pays = list(corpus["#3 country_label"])
list_province = list(corpus["#4 province_label"])


# In[13]:


#nétoyage des tweets (URL, hashtags, emoticones/emojis, ponctuation, translittération de arabizi, bruits)
list_enonce_clean = []
for ligne in list_enonce:
    ligne = araby.strip_tashkeel(ligne)   #Supprimer les signes diacritique (voyelles)
    ligne = araby.normalize_hamza(ligne)  #Normalise une chaîne (hamza ou les différents A en arabe)
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)
    ligne = re.sub(emoji_pattern, '', ligne)
    ligne = buckwalter.untransliterate(ligne) # translittération du L'arabizi
    ligne = re.sub(r"(http|https|ftp)://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+","",ligne) # supprimer les URL
    ligne = re.sub(r"[أ-ي]#","",ligne)
    ligne = re.sub(r"@[^\s]+","",ligne) # suprimer les hashtags et les noms d'utilisateur
    ligne = re.sub(r"#[أ-ي]+","",ligne)
    ligne = re.sub('[%s]' % re.escape(string.punctuation), '', ligne)
    ligne = re.sub(r"(\d|[\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669])+",'', ligne) # supprimer les nombres
    ligne = re.sub(r'[^\s\u0621\u0622\u0623\u0624\u0625\u0626\u0627\u0628\u0629\u062A\u062B\u062C\u062D\u062E\u062F\u0630\u0631\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063A\u0640\u0641\u0642\u0643\u0644\u0645\u0646\u0647\u0648\u0649\u064A]','',ligne) # supprimer les mots non-arabe
    ligne = re.sub(r'\s+',' ',ligne) # Supprimer les doubles espaces 
    ligne = re.sub(r'[^\u0600-\u06FF]', ' ', ligne) # supprimer les symboles non-arabe
    list_enonce_clean.append(ligne) 
list_enonce_clean


# In[18]:


# segmentation et restructuratuion des données (de manière a avoir un token par ligne)
list_token = []
list_ID_enonce = []
list_ID_token = []
list_p = []
list_v = []
list_len = []
list_e = []
ID_enonce = 0

for enonce, pays, province in zip(list_enonce_clean, list_pays, list_province):
        ID_token = 0
        ID_enonce = ID_enonce
        ligne = araby.tokenize(enonce1) # tokenisation des enoncés 
        for token in ligne:
            ID_token = ID_token+1
            list_ID_enonce.append(str(ID_enonce))
            list_ID_token.append(str(ID_token))
            list_token.append(token)
            list_p.append(pays)
            list_v.append(province)
            list_len.append(len(token))
            list_e.append(enonce)
        ID_enonce = ID_enonce+1


# In[19]:


# réintégration des données dans un nouveau DataFrame
df = pd.DataFrame(columns=['#ID_ENONCE','#ID_TOKEN','#LEN_TOKEN','#TOKEN','#PAYS','#PROVINCE','#ENONCE'])
df['#ID_ENONCE'] =list_ID_enonce           
df['#ID_TOKEN'] = list_ID_token
df['#TOKEN'] = list_token
df['#PAYS'] =  list_p
df['#PROVINCE'] = list_v
df['#LEN_TOKEN'] = list_len
df['#ENONCE'] =list_e     
df


# In[20]:


df.to_csv("CorpusARD.csv", index = False)


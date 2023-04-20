#%%
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from nltk.corpus import stopwords
from collections import Counter
import string
import re
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%%
df = pd.read_csv('data/imdb.csv')

# Splitting to train and test data
x, y = df['review'].values, df['sentiment'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)

#%%

# Analysis sentiment

dd = pd.Series(y_train).value_counts()
sns.barplot(x=np.array(['negative', 'positive']), y=dd);

#%%

# Tokenization

def preprocess_string(s):
  # Remove all non-word characters (everything except numbers and letters)
  s = re.sub(r"[^\w\s]", '', s)
  # Replace all runs of whitespaces with no space
  s = re.sub(r"\s+", '', s)
  # replace digits with no space
  s = re.sub(r"\d", '', s)
  return s

def finalize(data, vocab):
  #final_list = []
  #for sen in data:
  #  final_list.append([vocab[preprocess_string(word)]
  #                            for word in sen.lower().split()
  #                            if preprocess_string(word) in vocab.keys()])

  final_list = [[vocab[preprocess_string(word)]
    for word in sen.lower().split()
    if preprocess_string(word) in vocab.keys()] for sen in data]

  return final_list

def tokenize(x_train, y_train, x_test, y_test):
  sw = set(stopwords.words('english'))
  word_list = [word for sen in x_train for word in sen.lower().split() if word not in sw and word != '']

  corpus = Counter(word_list)
  # sort by most common words; pick 1000 most common
  corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]

  vocab = {w:i+1 for i, w in enumerate(corpus_)}

  final_list_train = finalize(x_train, vocab)
  final_list_test = finalize(x_test, vocab)

  encoded_train = np.array([1 if y == 'positive' else 0 for y in y_train])
  encoded_test = np.array([1 if y == 'positive' else 0 for y in y_test])
  return final_list_train, encoded_train, final_list_test, encoded_test, vocab

#%%
x_train, y_train, x_test, y_test, vocab = tokenize(x_train, y_train, x_test, y_test)
#%%

rev_len = [len(i) for i in x_train]
pd.Series(rev_len).describe()
#pd.Series(rev_len).hist();

# This Python 3 environment
import gc
import os
import logging
import datetime
import warnings
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import itertools 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#plt.style.use('seaborn-white')
#plt.style.use('seaborn')
import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

# loading data in pandas df
df = pd.read_csv('pima.csv')
df.info()

diabetic=df[df['Outcome']==1]  #base de donnees des diabetic
nondiabetic=df[df['Outcome']==0] # base desnon diabetics
df.isnull().sum()

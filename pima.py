# This Python 3 environment
import warnings
import numpy as np # Algebre lineaire 
import pandas as pd # lecture des donnees en CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import itertools 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#plt.style.use('seaborn-white')
#plt.style.use('seaborn')
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+- #
# On charge les donnees avec pandas.
df = pd.read_csv('pima.csv')
df.info()

df.isnull().sum() # Pas de donnees manquantes

diabetic=df[df['Outcome']==1]  # les personne diabetiques
nondiabetic=df[df['Outcome']==0] # les personnes non diabetics

# Distribution de la variables Outcome dans le base

sns.countplot(x='Outcome',data=df)
plt.show()
df['Outcome'].value_counts(1)  #len(df[df.Outcome == 1])/len(df['Outcome'])
# Nous avons plus de individus sains.



# Correlation
corr = df.corr()
corr

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+- #

# Randomiser les donnees 
from sklearn.utils import shuffle

# Echantillon de train et  de test
seed = 123
X_train, X_test= train_test_split(df, test_size =0.25, random_state=seed)

app_X  = X_train[X_train.columns[:8]]
test_X = X_test[X_test.columns[:8]]
app_Y  = X_train['Outcome']
test_Y = X_test['Outcome']


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+- #

# 1 ! Neural network

import keras 
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense
from keras.utils.vis_utils import plot_model

# Definisons les callback sur la base de loss pour l'arrete de l'entrainement
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.15):
            print("\n Arret de l'entrainement avec car car loss est inferieur a .15")
            self.model.stop_training = True
          
          
# 2. Définir le modèle
# Les modèles de Keras sont définis comme une séquence de couches. Nous créons un modèle séquentiel et ajoutons des couches  
# une par une jusqu'à ce que notre topologie de réseau nous satisfasse soit 12, 8, 8, 8, 1.  


# create m
m = Sequential()
m.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
m.add(Dense(8, init='uniform', activation='relu'))
m.add(Dense(8, init='uniform', activation='relu'))
m.add(Dense(8, init='uniform', activation='relu'))
m.add(Dense(1, init='uniform', activation='sigmoid'))
m.summary()

# 3. compilation et execution du modele
m.compile(loss='mean_squared_error' , optimizer='adam', metrics=['accuracy'])
callbacks = myCallback()
history = m.fit(app_X, app_Y, epochs=400, batch_size=len(app_X), verbose=1, callbacks=[callbacks])

# le courbe de l'accuracy
plt.plot(history.history['acc'])
plt.show()

# Evaluation du modele

# Evaluation du modele
scores = m.evaluate(app_X, app_Y)
print("\n%s: %.2f%%" % (m.metrics_names[1], scores[1]*100))

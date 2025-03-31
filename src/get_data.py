import pandas as pd
import numpy as np
import sklearn
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
#Les messages d'erreurs d'importation de tensorflow.keras ne sont pas à prendre en compte : 
# l'importation est fait "parésseusement" donc l'appelse fait seulement lorsque nous avons besoin des imports

link1 = "./data/MusicGenre/features_3_sec.csv"
link2 = "./data/MusicGenre/features_30_sec.csv"
#Nous ne travaillerons pas avec ces données pour le moment puisqu'elle représente les données de manière générale

def get_data(link):
    data = pd.read_csv(link)
    return data

data=get_data(link1)
#print(data)

def data_treatment (data):
    #On enregistre les labels dans une variable extérieur
    label = data['label']
    data = data.drop(columns=['label'])
    
    data_to_drop=['filename', 'length', 'rms_mean', 'rms_var', 'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo']
    data = data.drop(columns=data_to_drop)
    
    return data, label

data, label=data_treatment(data)
#print(data)

'''
Les données sur 30 secondes correspondent à toutes les données
Les données toutes les 3 secondes seront celles à utiliser puisqu'elle décomposent les valeurs des features en 9 parties
temporatilté sur 9 parties de 3 secondes 
'''
def data_preparation(data, label):
    # On prépare les données sous forme de matrices pour pouvoir les enregistrées dans notre tableau numpy
    liste_matrice = []
    matrice = np.array([])
    
    for i in range(0, len(data)):
        #Je récupère les valeurs de chaque ligne et je les mets dans une matrice
        matrice = np.array([data.iloc[i].values for _ in range(10)])
        liste_matrice.append(matrice)
    
    #On enregistre tout au format numpy 
    final_matrice = np.array(liste_matrice)

    print(label)
    encoder = LabelEncoder()
    label = encoder.fit_transform(label)  # Convertit les catégories en entiers


    return final_matrice, label

'''
Les données doivent être de la forme (9990, 10, 46) afin de pouvoir effectuer le train_test_split
Pour pouvoir avoir les dimensions 9990 en profondeur, nous avons juste à prendre la transposée : final_matrice.T
'''
data, label = data_preparation(data, label)
print(data.shape)
# On split de manière aléatoire nos données : 
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        data, label, test_size=0.2, random_state=42)

def RNN():
    # Création du modèle RNN : 
    model = Sequential()
    model.add(keras.layers.SimpleRNN(64,input_shape=(10, 46), return_sequences=False))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()
    
    model.compile(loss="binary_crossentropy",
                  optimizer='adam', metrics=['accuracy'])

    print("Training RNN...")
    print(X_train.dtype)  # Devrait être float32 ou float64
    print(Y_train.dtype)
    
    history = model.fit(X_train, 
                        Y_train, 
                        validation_data=(X_test, Y_test), 
                        batch_size=5 
                        )

    
RNN()


def LSTM():
    # Création du modèle RNN : 
    model = Sequential()
    model.add(keras.layers.LSTM(64,input_shape=(10, 46), return_sequences=False))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()
    
    model.compile(loss="binary_crossentropy",
                  optimizer='adam', metrics=['accuracy'])

    print("Training LSTM...")
    print(X_train.dtype)  # Devrait être float32 ou float64
    print(Y_train.dtype)
    
    history = model.fit(X_train, 
                        Y_train, 
                        validation_data=(X_test, Y_test), 
                        batch_size=5 
                        )

    
LSTM()
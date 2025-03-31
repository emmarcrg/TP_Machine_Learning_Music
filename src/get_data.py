from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

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
encoder = LabelEncoder()
def data_preparation(data, label):
    # On prépare les données sous forme de matrices pour pouvoir les enregistrées dans notre tableau numpy
    liste_matrice = []
    matrice = np.array([])
    
    # Initialisation du scaler
    scaler = StandardScaler()

    # Normalisation des données
    data_normalized = scaler.fit_transform(data)
    data = pd.DataFrame(data_normalized)
    
    for i in range(0, len(data)):
        #Je récupère les valeurs de chaque ligne et je les mets dans une matrice
        matrice = np.array([data.iloc[i].values for _ in range(10)])
        liste_matrice.append(matrice)
    
    #On enregistre tout au format numpy 
    final_matrice = np.array(liste_matrice)

    #Mettre l'ensemble des données des labels au format entier
    label = encoder.fit_transform(label)  

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

def RNN( nb_neuronnes : int, nb_couches : int, optimiseur : str, fonction_activation : str, batch_size : float):
    # Création du modèle RNN : 
    model = Sequential()
    ratio = max(1, nb_neuronnes // nb_couches)
    ratio= int(ratio)
    print(f"Le ratio est de : {ratio}")
    
    model.add(keras.layers.SimpleRNN(nb_neuronnes,input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    
    neuronnes = nb_neuronnes
    for i in range(nb_couches - 1):
        if neuronnes-ratio <= 0:  # Vérifiez que le nombre de neurones reste valide
            print("Nombre de neurones trop faible, ajustement à 1.")
            neuronnes = neuronnes
            pass
        else:
            neuronnes -= ratio
            model.add(keras.layers.SimpleRNN(neuronnes, return_sequences=True))    

    
    # Dernière couche RNN avec return_sequences=False
    if neuronnes-ratio <= 0:
        neuronnes = neuronnes - int(neuronnes/2)
    else :
        neuronnes -= ratio
    model.add(keras.layers.SimpleRNN(neuronnes, return_sequences=False))
    model.add(keras.layers.Dense(10, activation=fonction_activation))
    
    model.summary()
    
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimiseur, metrics=['accuracy'])

    print(f"Training RNN avec l'optimiseur {optimiseur} et la fonction d'activation {fonction_activation}...")
    
    history = model.fit(X_train, 
                        Y_train, 
                        validation_data=(X_test, Y_test), 
                        batch_size=batch_size)
    
    return model


# Test optimiseurs et fonctions d'activation :
'''print("Test des optimiseurs et fonction d'activation ")
RNN(nb_neuronnes = 46, nb_couches = 4, optimiseur='adam', fonction_activation='tanh', batch_size= 5)
RNN(nb_neuronnes = 46, nb_couches = 4, optimiseur = 'RMSprop', fonction_activation = 'tanh', batch_size= 5)

RNN(nb_neuronnes = 46, nb_couches = 4, optimiseur='adam', fonction_activation='softmax', batch_size = 5)
RNN(nb_neuronnes = 46, nb_couches = 4, optimiseur = 'RMSprop', fonction_activation = 'softmax', batch_size = 5)'''

# Test nombre de neuronnes et nombre de couches : en se basant sur les meilleures performances passées 
'''print("Test du nombre de neuronnes et de couches ")
RNN(nb_neuronnes = 128, nb_couches = 4, optimiseur='adam', fonction_activation='softmax', batch_size = 5)
RNN(nb_neuronnes = 128, nb_couches = 8, optimiseur='adam', fonction_activation='softmax', batch_size = 5)
RNN(nb_neuronnes = 62, nb_couches = 4, optimiseur='adam', fonction_activation='softmax', batch_size = 5)
RNN(nb_neuronnes = 62, nb_couches = 8, optimiseur='adam', fonction_activation='softmax', batch_size = 5)'''

# Test du batch size avec celui qui avait fonctionné le mieux :
'''print("Test des batch size ")
RNN(nb_neuronnes = 46, nb_couches = 4, optimiseur='adam', fonction_activation='softmax', batch_size = 10)
RNN(nb_neuronnes = 46, nb_couches = 4, optimiseur='adam', fonction_activation='softmax', batch_size = 20)
RNN(nb_neuronnes = 46, nb_couches = 4, optimiseur='adam', fonction_activation='softmax', batch_size = 100)'''

# Test croisé 
#RNN(nb_neuronnes = 250, nb_couches = 25, optimiseur='adam', fonction_activation='softmax', batch_size = 20)

def LSTM(nb_neuronnes : int, nb_couches : int, optimiseur : str, fonction_activation : str, batch_size : float):
    # Création du modèle RNN : 
    model = Sequential()
    ratio = max(1, nb_neuronnes // nb_couches)
    ratio= int(ratio)
    print(f"Le ratio est de : {ratio}")
    
    model.add(keras.layers.LSTM(nb_neuronnes,input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    
    neuronnes = nb_neuronnes
    for i in range(nb_couches - 1):
        if neuronnes-ratio <= 0:  # Vérifiez que le nombre de neurones reste valide
            print("Nombre de neurones trop faible, ajustement à 1.")
            neuronnes = neuronnes
            pass
        else:
            neuronnes -= ratio
            model.add(keras.layers.LSTM(neuronnes, return_sequences=True))    

    
    # Dernière couche RNN avec return_sequences=False
    if neuronnes-ratio <= 0:
        neuronnes = neuronnes - int(neuronnes/2)
    else :
        neuronnes -= ratio
    model.add(keras.layers.LSTM(neuronnes, return_sequences=False))
    model.add(keras.layers.Dense(10, activation=fonction_activation))
    
    model.summary()
    
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimiseur, metrics=['accuracy'])

    print(f"Training LSTM avec l'optimiseur {optimiseur} et la fonction d'activation {fonction_activation}...")
    
    history = model.fit(X_train, 
                        Y_train, 
                        validation_data=(X_test, Y_test), 
                        batch_size=batch_size)
    
    return model

# Test optimiseurs et fonctions d'activation :
'''print("Test des optimiseurs et fonctions d'activation ")
LSTM(nb_neuronnes = 46, nb_couches = 4, optimiseur='adam', fonction_activation='tanh', batch_size= 5)
LSTM(nb_neuronnes = 46, nb_couches = 4, optimiseur = 'RMSprop', fonction_activation = 'tanh', batch_size= 5)

LSTM(nb_neuronnes = 46, nb_couches = 4, optimiseur='adam', fonction_activation='softmax', batch_size = 5)
LSTM(nb_neuronnes = 46, nb_couches = 4, optimiseur = 'RMSprop', fonction_activation = 'softmax', batch_size = 5)'''

# Test nombre de neuronnes et nombre de couches : en se basant sur les meilleures performances passées 
'''print("Test du nombre de neuronnes et des couches ")
LSTM(nb_neuronnes = 128, nb_couches = 4, optimiseur='adam', fonction_activation='softmax', batch_size = 5)
LSTM(nb_neuronnes = 128, nb_couches = 8, optimiseur='adam', fonction_activation='softmax', batch_size = 5)
LSTM(nb_neuronnes = 62, nb_couches = 4, optimiseur='adam', fonction_activation='softmax', batch_size = 5)
LSTM(nb_neuronnes = 62, nb_couches = 8, optimiseur='adam', fonction_activation='softmax', batch_size = 5)

# Test du batch size avec celui qui avait fonctionné le mieux :
print("Test des batch size ")
LSTM(nb_neuronnes = 46, nb_couches = 4, optimiseur='adam', fonction_activation='softmax', batch_size = 10)
LSTM(nb_neuronnes = 46, nb_couches = 4, optimiseur='adam', fonction_activation='softmax', batch_size = 20)
LSTM(nb_neuronnes = 46, nb_couches = 4, optimiseur='adam', fonction_activation='softmax', batch_size = 100)'''

def analyse_erreur_RNN ():
    RNN_model = RNN(nb_neuronnes = 46, nb_couches = 4, optimiseur='adam', fonction_activation='softmax', batch_size = 5)
    y_pred = RNN_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    mc = confusion_matrix(Y_test, y_pred_classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=mc, display_labels=encoder.classes_)
    cm_display.plot()
    plt.show()
    
def analyse_erreur_LSTM ():
    RNN_model = LSTM(nb_neuronnes = 46, nb_couches = 4, optimiseur='adam', fonction_activation='softmax', batch_size = 5)
    y_pred = RNN_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    mc = confusion_matrix(Y_test, y_pred_classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=mc, display_labels=encoder.classes_)
    cm_display.plot()
    plt.show()
    
analyse_erreur_RNN()
analyse_erreur_LSTM()
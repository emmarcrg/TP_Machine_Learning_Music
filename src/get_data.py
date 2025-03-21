import pandas as pd
import numpy as np

link1 = "./data/MusicGenre/features_3_sec.csv"
link2 = "./data/MusicGenre/features_30_sec.csv"

def get_data(link):
    data = pd.read_csv(link)
    return data

data1=get_data(link1)
#print(data1)

data2=get_data(link2)
#print(data2)

def data_treatment (data):
    #On enregistre les labels dans une variable extérieur
    label = data['label']
    data = data.drop(columns=['label'])
    
    data_to_drop=['filename', 'length', 'rms_mean', 'rms_var', 'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo']
    data = data.drop(columns=data_to_drop)
    
    return data

data1=data_treatment(data1)
#print(data1)
data2=data_treatment(data2)
#print(data2)

'''
Les données sur 30 secondes correspondent à toutes les données
Les données toutes les 3 secondes seront celles à utiliser puisqu'elle décomposent les valeurs des features en 9 parties
temporatilté sur 9 parties de 3 secondes 
'''

def data_preparation():
    # On prépare les données sous forme de matrices pour pouvoir les enregistrées dans notre tableau numpy
    liste_matrice = []
    matrice = np.array([])
    
    print(len(data1))
    for i in range(0, len(data1)):
        #Je récupère les valeurs de chaque ligne et je les mets dans une matrice
        matrice = np.array([data1.iloc[i].values for _ in range(10)])
        liste_matrice.append(matrice)
    
    #On enregistre tout au format numpy 
    final_matrice = np.array(liste_matrice)
    return final_matrice.T
    
data = data_preparation()
print(data)
print(data.shape)
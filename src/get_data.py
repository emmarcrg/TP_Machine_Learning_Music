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
    
    data_to_drop=[ 'rms_mean', 'rms_var', 'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo']
    data = data.drop(columns=data_to_drop)
    
    return data

data1=data_treatment(data1)
print(data1)
data2=data_treatment(data2)
print(data2)

'''
Les données sur 30 secondes correspondent à toutes les données
Les données toutes les 3 secondes seront celles à utiliser puisqu'elle décomposent les valeurs des features en 9 parties
temporatilté sur 9 parties de 3 secondes 
'''

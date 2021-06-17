

from tensorflow.python.framework.error_interpolation import parse_message
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.manifold import TSNE as tsne

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pandas as pd
import os
import random
from scipy import stats
from pylab import rcParams
from nhis_autoencoder import nhis_encoder, nhis_decoder, nhis_autoencoder



class nhis_clustering():

    def __init__(self, data_path: str, num_layers: int, encoder_input_shape: tuple, decoder_input_shape: tuple) -> None:
        self.data = pd.read_csv(data_path, encoding= "euc-kr", header= None, dtype= None)
        self.useful_col = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19] # 신장(5Cm단위),체중(5Kg단위),허리둘레,시력(좌),시력(우),수축기혈압,이완기혈압,식전혈당(공복혈당),총콜레스테롤,트리글리세라이드,HDL콜레스테롤,LDL콜레스테롤,혈색소,혈청크레아티닌,(혈청지오티)AST,(혈청지오티)ALT,감마지티피																	
        self.nhis_autoencoder = nhis_autoencoder( 3, encoder_input_shape, decoder_input_shape )
        

    
    def data_preprocessing(self, valid_size: int) -> np.ndarray:
        raw_data = self.data.values[1:,]
        raw_data = raw_data[:,5:25].astype(np.float64)
        used_data = raw_data[:,self.useful_col]
        
        print(np.count_nonzero(~np.isnan(used_data).any(axis=1))) #996603
        used_data = used_data[~np.isnan(used_data).any(axis=1)]
        
        #16 features + choleterol groups
        data_final = np.zeros((996603,17)) 
        data_final[:,0] = np.round(used_data[:,1]/((used_data[:,0]/100)**2)) # BMI formula
        data_final[:,1:-1] = used_data[:,2:]                                 # copy the rest
        data_final[:,-1] = used_data[:,8]                                    # dummy for the label later

        # labeling
        chl_idx1 = np.where(data_final[:,-1] < 125)[0] # 20502
        chl_idx2 = np.where(data_final[:,-1] > 280)[0] # 20501
        chl_idx3 = np.where(np.logical_and(data_final[:,-1] > 175, data_final[:,-1] < 178))[0] # 20541
        chl_idx = np.concatenate([chl_idx1, chl_idx2, chl_idx3]) # 61544

        chl_idx = np.sort(chl_idx) 
        data_final[chl_idx1,-1] = 1 # low
        data_final[chl_idx2,-1] = 2 # high
        data_final[chl_idx3,-1] = 3 # normal

        data_final = data_final[chl_idx, :]            # only use the ones in one of the three groups 

        np.random.seed(0)                              # to get the same result and compare

        perm_list = list(np.random.permutation(len(chl_idx)))
        rnd_train = perm_list[:len(chl_idx)-valid_size]
        rnd_valid = perm_list[len(chl_idx)-valid_size:]

        train_data = data_final[rnd_train, :-1]       # make sure the label itself is not included
        valid_data = data_final[rnd_valid, :-1]       # make sure the label itself is not included

        valid_groups = data_final[rnd_valid, -1]      # the label of validation data
        unique_groups = np.unique(valid_groups)       # the unqiue labels of validation data -> 1,2,3
     
        #train_data = preprocessing.normalize(train_data, norm= 'l2') #nomalization using sklearn
        #valid_data = preprocessing.normalize(valid_data, norm= 'l2') #nomalization using sklearn
        normalizer = Normalizer().fit(train_data)
        train_data = normalizer.transform(train_data)
        valid_data = normalizer.transform(valid_data)

        return train_data, valid_data, valid_groups, unique_groups



    

    def model_training(self):
        pass
    
    


nhis_c = nhis_clustering("./NHIS_OPEN_GJ_2017.csv", 3, (16,), (13,))
train_data, valid_data, valid_groups, unique_groups = nhis_c.data_preprocessing(400)

print(train_data[:5,:])
print(valid_data[:5,:])
print(valid_groups[:5])
print(unique_groups[:5])

#nhis_c.data_preprocessing(3)
    
#from tensorflow.python.framework.error_interpolation import parse_message
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.manifold import TSNE as tsne
from sklearn.cluster import KMeans
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pandas as pd
import os
import random
from scipy import stats
from pylab import rcParams
from nhis_autoencoder import NhisAutoencoder, NhisEncoder, NhisDecoder
from nhis_vae import VariationalAutoEncoder, Encoder, Decoder
import shutil
import plotly.express as px
from copy import deepcopy





class NhisClustering():

    def __init__(self, data_path: str,  original_dim: int, latent_dim: int) -> None:
        self.data = pd.read_csv(data_path, encoding="cp949")
        self.original_dim = original_dim
        self.latent_dim   = latent_dim
        self.useful_col = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 16, 17, 18, 19] # 신장(5Cm단위),체중(5Kg단위),허리둘레,시력(좌),시력(우),수축기혈압,이완기혈압,식전혈당(공복혈당),총콜레스테롤,트리글리세라이드,HDL콜레스테롤,LDL콜레스테롤,혈색소,혈청크레아티닌,(혈청지오티)AST,(혈청지오티)ALT,감마지티피																	
        self.nhis_autoencoder = NhisAutoencoder( self.original_dim, self.latent_dim )
        self.nhis_vae = VariationalAutoEncoder( self.original_dim, self.latent_dim )
        self.kmc      = KMeans(latent_dim)

    
    def data_preprocessing(self, valid_size: int) -> np.ndarray:
        raw_data = self.data.values[1:,] # dropping the column names
        raw_data = raw_data[:,5:25].astype(np.float64)
        used_data = raw_data[:,self.useful_col]
        
        print(np.count_nonzero(~np.isnan(used_data).any(axis=1))) #996603
        used_data = used_data[~np.isnan(used_data).any(axis=1)]
        
        #13 features + choleterol groups
        data_final = np.zeros((np.count_nonzero(~np.isnan(used_data).any(axis=1)), 15)) 
        data_final[:,0] = np.round(used_data[:,1]/((used_data[:,0]/100)**2)) # BMI formula
        data_final[:,1:-1] = used_data[:,2:]                                 # copy the rest
        data_final[:,-1] = used_data[:,8]                                    # dummy for the label later
        data_final = np.delete(data_final, 8, 1)                             # drop the cholesterol column    
        
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

        normalizer = Normalizer().fit(train_data)
        train_data = normalizer.transform(train_data)
        valid_data = normalizer.transform(valid_data)
    
        return train_data, valid_data, valid_groups, unique_groups


    def model_training(self, train_data, epoch, model_type):
        if model_type == 'vae':
            path = './training/VAE'
            autoencoder = self.nhis_vae

        elif model_type == 'ae':
            path = './training/AE'
            autoencoder = self.nhis_autoencoder

        autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsolutePercentageError())

        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok = True)
        
        
        es = EarlyStopping(monitor='val_loss', mode = 'min' , patience = 15, verbose = 1)
        file_path = path + '/Epoch_{epoch:03d}_Val_{val_loss:.3f}.hdf5'
        mc = ModelCheckpoint(file_path, monitor='val_loss', mode='min',verbose=1, \
                             save_best_only=True, save_weights_only=True)

        print('\n \t\t\t #################### model fitting starts #################### \t\t\t \n')

        hist = autoencoder.fit(train_data, train_data,
                        epochs = epoch,
                        batch_size = 32,
                        shuffle = True,
                        validation_data = (valid_data, valid_data),
                        callbacks= [mc, es])
        
        return hist
        
    
    def tsne_clustering(self, train_data, valid_data):
        tsne_cluster2 = tsne(n_components = 2)
        tsne_cluster3 = tsne(n_components = 3)
        tsne_cluster2.fit(train_data)
        tsne_cluster3.fit(train_data)
        latent_vector2 = tsne_cluster2.fit_transform(valid_data)
        latent_vector3 = tsne_cluster3.fit_transform(valid_data)

        return latent_vector3, latent_vector2

    def cluster_visualization_3D(self, latent_vector, valid_groups, unique_groups):
        low_idx = np.where(valid_groups == 1)[0]
        high_idx = np.where(valid_groups == 2)[0]
        no_idx = np.where(valid_groups == 3)[0]

        xs = latent_vector[:, 0]
        ys = latent_vector[:, 1]
        zs = latent_vector[:, 2]

        # visualize in 3D plot
        rcParams['figure.figsize'] = 12, 10

        fig1 = plt.figure(1)

        ax1 = fig1.add_subplot(221, projection = '3d')
        ax2 = fig1.add_subplot(222, projection = '3d')
        ax3 = fig1.add_subplot(223, projection = '3d')
        ax4 = fig1.add_subplot(224, projection = '3d')

        #plot1 / ax1
        low_xs = latent_vector[low_idx, 0]
        low_ys = latent_vector[low_idx, 1]
        low_zs = latent_vector[low_idx, 2]
        for x, y, z in zip(low_xs, low_ys, low_zs):
            ax1.text(x, y, z, 1, fontsize = 4, backgroundcolor='red')
            
        ax1.set_xlim(xs.min()*2, xs.max()*2)
        ax1.set_ylim(ys.min()*2, ys.max()*2)
        ax1.set_zlim(zs.min()*2, zs.max()*2)

        #plot2 / ax2
        high_xs = latent_vector[high_idx, 0]
        high_ys = latent_vector[high_idx, 1]
        high_zs = latent_vector[high_idx, 2]
        for x, y, z in zip(high_xs, high_ys, high_zs):
            ax2.text(x, y, z, 2, fontsize = 4, backgroundcolor='blue')
            
        ax2.set_xlim(xs.min()*2, xs.max()*2)
        ax2.set_ylim(ys.min()*2, ys.max()*2)
        ax2.set_zlim(zs.min()*2, zs.max()*2)

        #plot3 / ax3
        no_xs = latent_vector[no_idx, 0]
        no_ys = latent_vector[no_idx, 1]
        no_zs = latent_vector[no_idx, 2]
        for x, y, z in zip(high_xs, high_ys, high_zs):
            ax3.text(x, y, z, 3, fontsize = 4, backgroundcolor='green')
            
        ax3.set_xlim(xs.min()*2, xs.max()*2)
        ax3.set_ylim(ys.min()*2, ys.max()*2)
        ax3.set_zlim(zs.min()*2, zs.max()*2)

        #plot4 / ax4
        color=['red','blue', 'green'] 

        for x, y, z, label in zip(xs, ys, zs, valid_groups):
            col_ind = np.where(unique_groups == label)[0][0]
            c = color[col_ind]
            ax4.text(x, y, z, label, fontsize = 4, backgroundcolor=c)
            
        ax4.set_xlim(xs.min()*2, xs.max()*2)
        ax4.set_ylim(ys.min()*2, ys.max()*2)
        ax4.set_zlim(zs.min()*2, zs.max()*2)

        plt.savefig('3D_3groups_CHL.png', dpi=100)
        plt.show()


    def cluster_visualization_2D(self, latent_vector2, valid_groups, unique_groups):
        low_idx = np.where(valid_groups == 1)[0]
        high_idx = np.where(valid_groups == 2)[0]
        no_idx = np.where(valid_groups == 3)[0]

        xs = latent_vector2[:, 0]
        ys = latent_vector2[:, 1]
        
        # visualize in 2D plot
        rcParams['figure.figsize'] = 10, 8
        fig2 = plt.figure(2)

        ax1 = fig2.add_subplot(221)
        ax2 = fig2.add_subplot(222)
        ax3 = fig2.add_subplot(223)
        ax4 = fig2.add_subplot(224)
        
        # plot1 / ax1
        low_xs = latent_vector2[low_idx, 0]
        low_ys = latent_vector2[low_idx, 1]

        ax1.scatter(low_xs, low_ys, s = 40 , color = 'red', label = 'low cholesterol')
        ax1.set_xlim(xs.min()*2, xs.max()*2)
        ax1.set_ylim(ys.min()*2, ys.max()*2)

        # plot2 / ax2
        high_xs = latent_vector2[high_idx, 0]
        high_ys = latent_vector2[high_idx, 1]

        ax2.scatter(high_xs, high_ys, s = 40 , color = 'blue', label = 'high cholesterol')
        ax2.set_xlim(xs.min()*2, xs.max()*2)
        ax2.set_ylim(ys.min()*2, ys.max()*2)

        # plot3 / ax3
        no_xs = latent_vector2[no_idx, 0]
        no_ys = latent_vector2[no_idx, 1]

        ax3.scatter(no_xs, no_ys, s = 40 , color = 'green', label = 'normal cholesterol')
        ax3.set_xlim(xs.min()*2, xs.max()*2)
        ax3.set_ylim(ys.min()*2, ys.max()*2)
        
        # plot4 / ax4
        color = ['red', 'blue', 'green'] 
        labels = ['low cholesterol', 'high cholesterol', 'normal cholesterol']
        
        ax4.scatter(low_xs, low_ys, s = 40 , color = 'red', label = 'low cholesterol')
        ax4.scatter(high_xs, high_ys, s = 40 , color = 'blue', label = 'high cholesterol')
        ax4.scatter(no_xs, no_ys, s = 40 , color = 'green', label = 'normal cholesterol')
        ax4.set_xlim(xs.min()*2, xs.max()*2)
        ax4.set_ylim(ys.min()*2, ys.max()*2)

        ax1.legend(prop={'size': 8})
        ax2.legend(prop={'size': 8})
        ax3.legend(prop={'size': 8})
        ax4.legend(prop={'size': 8})
        plt.savefig('2D_3groups_CHL.png')
        plt.show()
    
    def load_model(self, path, model_type):
        if model_type == 'vae':
            self.nhis_vae(tf.ones((1, self.original_dim)))
            self.nhis_vae.load_weights(path)
            encoder = self.nhis_vae.encoder
        elif model_type == 'ae':
            self.nhis_autoencoder(tf.ones((1, self.original_dim)))
            self.nhis_autoencoder.load_weights(path)
            encoder = self.nhis_autoencoder.encoder
        
        return encoder

    def plotting(self, latent_vector3, groups):
        # Applying t-sne to the 3-dimensional latent_vector
        latent_vector2 = tsne(n_components = 2).fit_transform(latent_vector3)

        low_idx = np.where(groups == 1)[0]
        high_idx = np.where(groups == 2)[0]
        no_idx = np.where(groups == 3)[0]

        df_3d = pd.DataFrame(latent_vector3, columns= ['x','y','z'])
        df_3d['groups'] = -1
        df_3d['groups'].iloc[low_idx] = 'low cholesterol'
        df_3d['groups'].iloc[high_idx] = 'high cholesterol'
        df_3d['groups'].iloc[no_idx] = 'normal cholesterol'
        
        fig = px.scatter_3d(df_3d, x='x', y='y', z='z', color='groups')
        fig.write_html('./3D_3groups_CHL.html')
        fig.show()

        df_2d = pd.DataFrame(latent_vector2, columns= ['x','y'])
        df_2d['groups'] = -1
        df_2d['groups'].iloc[low_idx] = 'low cholesterol'
        df_2d['groups'].iloc[high_idx] = 'high cholesterol'
        df_2d['groups'].iloc[no_idx] = 'normal cholesterol'
        
        fig = px.scatter(df_3d, x='x', y='y', color='groups')
        fig.write_html('./2D_3groups_CHL.html')
        fig.show()

    def get_kmeans_groups(self, latent_vector3):
        self.kmc.fit(latent_vector3)
        kmc_groups = self.kmc.predict(latent_vector3) + 1

        return kmc_groups
    
    def get_mean_values(self, latent_vector3, groups):
        low_idx = np.where(groups == 1)[0]
        high_idx = np.where(groups == 2)[0]
        no_idx = np.where(groups == 3)[0]

        low_means = latent_vector3[low_idx, :].mean(axis = 0)
        high_means = latent_vector3[high_idx, :].mean(axis = 0)
        no_means = latent_vector3[no_idx, :].mean(axis = 0)
        
        return [low_means, high_means, no_means]
    
    def get_misliad_num(self, latent_vector3, groups):
        mean_list = self.get_mean_values(latent_vector3, groups)
        
        mislaid_count = 0
        for i in range(latent_vector3.shape[0]):
            mean_lst = deepcopy(mean_list)
            data_point = latent_vector3[i]
            group = int(groups[i]) - 1
            
            group_mean = mean_lst.pop(group)
            other_mean1 = mean_lst.pop()
            other_mean2 = mean_lst.pop()

            assert len(mean_lst) == 0
            
            if abs(data_point - group_mean).sum() > abs(data_point - other_mean1).sum() or \
            abs(data_point - group_mean).sum() > abs(data_point - other_mean2).sum():
                mislaid_count += 1

        return mislaid_count


if __name__ == "__main__":
    nhis_c = NhisClustering("./NHIS_OPEN_GJ_2017.csv", 13 ,3)
    train_data, valid_data, valid_groups, unique_groups = nhis_c.data_preprocessing(400)
    #hist = nhis_c.model_training(train_data, 50, 'ae')

    encoder = nhis_c.load_model("./training/AE/Epoch_049_Val_11.092.hdf5", 'ae')
    # in order to see (visualize) how the data is distributed across the latent variables
    # get latent vector for visualization
    #latent_vector = encoder.predict(valid_data)
    #_, _, latent_vector3 = encoder(valid_data) # vae
    latent_vector3 = encoder(valid_data)        # ae
    
    latent_vector3 = latent_vector3.numpy() # an eager tensor is returned
    #kmc_groups  = nhis_c.get_kmeans_groups(latent_vector3)
    mislaid_count = nhis_c.get_misliad_num(latent_vector3, valid_groups)
    print(mislaid_count)

    nhis_c.plotting(latent_vector3, valid_groups)

    #latent_vector3, latent_vector2 = nhis_c.tsne_clustering(train_data, valid_data)
    #nhis_c.cluster_visualization_3D(latent_vector, valid_groups, unique_groups)
    #nhis_c.cluster_visualization_2D(latent_vector2, valid_groups, unique_groups)
    
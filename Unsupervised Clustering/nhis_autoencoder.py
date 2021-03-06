
from enum import auto
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python import training

class NhisEncoder(layers.Layer):
    
    def __init__(self, latent_dim):
        super(NhisEncoder, self).__init__()
        self.dense1 = Dense(128, activation = 'relu')
        #self.bn1 = BatchNormalization()
        self.dense2 = Dense(latent_dim, activation = 'relu')
        #self.bn2 = BatchNormalization()
    
    def call(self, inputs):
        x = self.dense1(inputs)
        #x = self.bn1(x)
        x = self.dense2(x)
        #x = self.bn2(x)
        return x


class NhisDecoder(layers.Layer):
    
    def __init__(self, original_dim):
        super(NhisDecoder,self).__init__()
        self.dense1 = Dense(128, activation = 'relu')
        #self.bn1 = BatchNormalization()
        self.dense2 = Dense(original_dim, activation = 'sigmoid')
        #self.bn2 = BatchNormalization()
        
    def call(self, inputs):
        x = self.dense1(inputs)
        #x = self.bn1(x)
        x = self.dense2(x)
        #x = self.bn2(x)
        return x



class NhisAutoencoder(tf.keras.Model):

    def __init__(self, original_dim, latent_dim):
        super(NhisAutoencoder, self).__init__()
        self.encoder = NhisEncoder(latent_dim) 
        self.decoder = NhisDecoder(original_dim)
        self.original_dim = original_dim

    def call(self, inputs):
        encoded       = self.encoder(inputs)
        reconstructed = self.decoder(encoded)
        return reconstructed

    def model(self):
        x = Input( shape = (self.original_dim, ))
        output = self.call(x)
        return Model(inputs = x, outputs = output, name = "autoencoder")

"""
nhis_auc = NhisAutoencoder(15, 3)
nhis_auc.model().summary()
nhis_auc.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
autoencoder = nhis_auc.model()


autoencoder.load_weights("./training/Epoch_002_Val_35.798.hdf5")
autoencoder.encoder.predict([[1,2,3]])
"""


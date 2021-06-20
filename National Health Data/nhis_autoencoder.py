
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class nhis_encoder(tf.keras.layers.Layer):
    
    def __init__(self, neurons, latent_dim):
        super(nhis_encoder, self).__init__()
        self.neurons = neurons
        for i in range(len(neurons)):
            vars(self)['encoded_{}'.format(i)] = Dense(self.neurons[i], activation = 'relu')
        
        #self.dense1 = Dense(128, activation = 'relu')
        #self.dense2 = Dense(latent_dim, activation = 'relu')
    
    def call(self, inputs):
        x = self.encoded_0(inputs)
        for i in range(1, len(self.neurons)):
            x = vars(self)['encoded_{}'.format(i)](x)

        #x = self.dense1(inputs)
        #x = self.dense2(x)

        return x


class nhis_decoder(tf.keras.layers.Layer):
    
    def __init__(self, original_dim):
        super(nhis_decoder,self).__init__()
        self.dense1 = Dense(128, activation = 'relu')
        self.dense2 = Dense(original_dim, activation = 'sigmoid')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)

        return x



class nhis_autoencoder(tf.keras.Model):

    def __init__(self, neurons, original_dim, latent_dim):
        super(nhis_autoencoder, self).__init__()
        self.encoder = nhis_encoder(neurons, latent_dim) 
        self.decoder = nhis_decoder(original_dim)
        self.original_dim = original_dim

    def call(self, inputs):
        encoded       = self.encoder(inputs)
        reconstructed = self.decoder(encoded)

        return reconstructed

    def model(self):
        x = Input( shape = (self.original_dim, ))
        output = self.call(x)
        return Model(inputs = x, outputs = output, name = "autoencoder")


nhis_auc = nhis_autoencoder([8, 3], 16, 3)
nhis_auc.model().summary()
nhis_auc.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())


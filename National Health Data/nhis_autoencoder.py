
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class nhis_encoder(tf.keras.Model):
    
    def __init__(self, repetitions, latent_dim, input_shape):
        super(nhis_encoder, self).__init__()
        self.repetitions = repetitions
        self._input_shape = input_shape
        self._latent_dim = latent_dim

        #encoder
        for i in range(self.repetitions):
            neurons = int(128 / (i + 1)) if i < 2 else self._latent_dim
            vars(self)['encoded{}'.format(i)] = Dense(neurons, activation= "relu")
            vars(self)['encoded{}_bn'.format(i)] =  BatchNormalization()

    
    def call(self, inputs):
        x = self.encoded0(inputs)
        x = self.encoded0_bn(x)
        x = self.encoded0_ac(x)

        #encoder
        for i in range(1, self.repetitions):
            x = vars(self)['encoded{}'.format(i)](x)        # dense layer
            x = vars(self)['encoded{}_bn'.format(i)](x)     # batch norm layer
            x = vars(self)['encoded{}_ac'.format(i)](x)     # relu activation
        
        return x
    
    def model(self):
        x = Input( shape= self._input_shape )
        return Model(inputs = x, outputs = self.call(x), name = "encoder")



class nhis_decoder(tf.keras.Model):
    
    def __init__(self, repetitions, input_shape):
        super(nhis_decoder,self).__init__()
        self.repetitions = repetitions
        self._input_shape = input_shape

        # decoder
        for i in range(self.repetitions):
            neurons = int(64 * (i + 1)) if i < 2 else 16
            vars(self)['decoded{}'.format(i)] = Dense(neurons)
            vars(self)['decoded{}_bn'.format(i)] = BatchNormalization()
            vars(self)['decoded{}_ac'.format(i)] = ReLU()
        
    def call(self, inputs):
        x = self.decoded0(inputs)
        x = self.decoded0_bn(x)
        x = self.decoded0_ac(x)

        #encoder
        for i in range(1, self.repetitions):
            x = vars(self)['decoded{}'.format(i)](x)        # dense layer
            x = vars(self)['decoded{}_bn'.format(i)](x)     # batch norm layer
            x = vars(self)['decoded{}_ac'.format(i)](x)     # relu activation
        
        return x

    def model(self):
        x = Input( shape = self._input_shape )
        return Model(inputs = x, outputs = self.call(x), name = "decoder")


class nhis_autoencoder(tf.keras.Model):

    def __init__(self, repetitions, encoder_input_shape, decoder_input_shape):
        super(nhis_autoencoder, self).__init__()
        self.encoder = nhis_encoder(repetitions, encoder_input_shape)
        self.decoder = nhis_decoder(repetitions, decoder_input_shape)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        reconstructed = self.decoder(encoded)

        return reconstructed


    def model(self):
        x = Input( shape= self.encoder._input_shape )
        output = self.call(x)
        return Model(inputs = x, outputs = output, name = "autoencoder")

"""
nhis_autoenc = nhis_autoencoder( 3, (16,), (3,) )
nhis_autoenc.encoder.model().summary()
nhis_autoenc.decoder.model().summary()
nhis_autoenc.model().summary()
"""

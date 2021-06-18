
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class nhis_encoder(tf.keras.Model):
    
    def __init__(self, neurons: list, input_shape: tuple) -> None :
        super(nhis_encoder, self).__init__()
        self.neurons     = neurons       # the last element should be the desired latent dimension
        self._input_shape = input_shape

        #encoder
        for i in range(len(self.neurons)): 
            num_neurons = self.neurons[i]
            vars(self)['encoded{}'.format(i)]    = Dense(num_neurons, activation= "relu")
            vars(self)['encoded{}_bn'.format(i)] = BatchNormalization()

    
    def call(self, inputs):
        x = self.encoded0(inputs)
        x = self.encoded0_bn(x)

        #encoder
        for i in range(1, len(self.neurons)):
            x = vars(self)['encoded{}'.format(i)](x)        # dense + relu activation layer
            x = vars(self)['encoded{}_bn'.format(i)](x)     # batch norm layer

        return x
    
    def model(self) -> Model:
        x = Input( shape= self._input_shape )
        return Model(inputs = x, outputs = self.call(x), name = "encoder")



class nhis_decoder(tf.keras.Model):
    
    def __init__(self, neurons: list, input_shape: tuple) -> None:
        super(nhis_decoder,self).__init__()
        self.neurons = neurons
        self._input_shape = input_shape

        # decoder
        for i in range(len(self.neurons)):
            num_neurons = self.neurons[i]
            vars(self)['decoded{}'.format(i)] = Dense(num_neurons, activation = 'relu')
            vars(self)['decoded{}_bn'.format(i)] = BatchNormalization()
        
    def call(self, inputs):
        x = self.decoded0(inputs)
        x = self.decoded0_bn(x)

        #encoder
        for i in range(1, len(self.neurons)):
            x = vars(self)['decoded{}'.format(i)](x)        # dense layer + relu activation
            x = vars(self)['decoded{}_bn'.format(i)](x)     # batch norm layer
        
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
nhis_enc = nhis_encoder( [128, 64, 32, 16, 3], (16,))
nhis_enc.model().summary()

nhis_dec = nhis_decoder( [16, 32, 64, 128, 16], (3,))
nhis_dec.model().summary()
"""
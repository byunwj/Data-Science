
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class nhis_encoder(tf.keras.layers.Layer):
    
    def __init__(self, neurons: list, input_shape: tuple) -> None :
        super(nhis_encoder, self).__init__()
        self.neurons      = neurons       # the last element should be the desired latent dimension
        self._input_shape = input_shape   

        self.flatten = Flatten() # for testing with mnist

        #encoder
        for i in range(len(self.neurons)): 
            num_neurons = self.neurons[i]
            vars(self)['encoded{}'.format(i)]    = Dense(num_neurons, activation= "relu")
            vars(self)['encoded{}_bn'.format(i)] = BatchNormalization()

    
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.encoded0(x)
        x = self.encoded0_bn(x)

        #encoder
        for i in range(1, len(self.neurons)):
            x = vars(self)['encoded{}'.format(i)](x)        # dense + relu activation layer
            x = vars(self)['encoded{}_bn'.format(i)](x)     # batch norm layer

        return x
    
    def model(self) -> Model:
        x = Input( shape= self._input_shape )
        return Model(inputs = x, outputs = self.call(x), name = "encoder")



class nhis_decoder(tf.keras.layers.Layer):
    
    def __init__(self, neurons: list, input_shape: tuple) -> None:
        super(nhis_decoder,self).__init__()
        self.neurons      = neurons
        self._input_shape = input_shape

        # decoder
        for i in range(len(self.neurons)):
            num_neurons = self.neurons[i]
            activation = None if i == len(self.neurons) -1 else 'relu'
            vars(self)['decoded{}'.format(i)]    = Dense(num_neurons, activation = activation)
            vars(self)['decoded{}_bn'.format(i)] = BatchNormalization()
        
    def call(self, inputs):
        x = self.decoded0(inputs)
        x = self.decoded0_bn(x)

        #encoder
        for i in range(1, len(self.neurons)):
            x = vars(self)['decoded{}'.format(i)](x)        # dense layer + relu activation
            x = vars(self)['decoded{}_bn'.format(i)](x)     # batch norm layer
        
        x = Reshape((28,28))(x)   # testing with mnist data
        return x

    def model(self):
        x = Input( shape = self._input_shape )
        return Model(inputs = x, outputs = self.call(x), name = "decoder")



class nhis_autoencoder(tf.keras.Model):

    def __init__(self, encoder_neurons: list, encoder_input_shape: tuple, decoder_neurons: list, decoder_input_shape: tuple) -> None :
        super(nhis_autoencoder, self).__init__()
        self.encoder = nhis_encoder( encoder_neurons , encoder_input_shape ) 
        self.decoder = nhis_decoder( decoder_neurons , decoder_input_shape )

    def call(self, inputs):
        encoded       = self.encoder(inputs)
        reconstructed = self.decoder(encoded)

        return reconstructed

    def model(self):
        x = Input( shape = self.encoder._input_shape )
        output = self.call(x)
        return Model(inputs = x, outputs = output, name = "autoencoder")


nhis_auc = nhis_autoencoder( [128, 64], (28,28,), [128, 784], (64,) )
#nhis_auc.encoder.model().summary()
#nhis_auc.decoder.model().summary()
nhis_auc.model().summary()

nhis_auc.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsolutePercentageError())

from tensorflow.keras.datasets import fashion_mnist

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print (x_train.shape)
print (x_test.shape)

"""
nhis_auc.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))
"""
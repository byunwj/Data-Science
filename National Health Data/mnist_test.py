
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class nhis_encoder(tf.keras.layers.Layer):
    
    def __init__(self):
        super(nhis_encoder, self).__init__()
        self.flatten = Flatten() # for testing with mnist
        self.dense1 = Dense(128, activation = 'relu')
        self.dense2 = Dense(64, activation = 'relu')
    
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


class nhis_decoder(tf.keras.layers.Layer):
    
    def __init__(self):
        super(nhis_decoder,self).__init__()
        self.dense1 = Dense(128, activation = 'relu')
        self.dense2 = Dense(784, activation = 'sigmoid')
        self.reshape = Reshape((28,28))
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.reshape(x)

        return x



class nhis_autoencoder(tf.keras.Model):

    def __init__(self):
        super(nhis_autoencoder, self).__init__()
        self.encoder = nhis_encoder() 
        self.decoder = nhis_decoder()

    def call(self, inputs):
        encoded       = self.encoder(inputs)
        reconstructed = self.decoder(encoded)

        return reconstructed

    def model(self):
        x = Input( shape = (28,28,))
        output = self.call(x)
        return Model(inputs = x, outputs = output, name = "autoencoder")


nhis_auc = nhis_autoencoder()
#nhis_auc.encoder.model().summary()
#nhis_auc.decoder.model().summary()
nhis_auc.model().summary()
nhis_auc.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

from tensorflow.keras.datasets import fashion_mnist

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print (x_train.shape)
print (x_test.shape)

nhis_auc.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))

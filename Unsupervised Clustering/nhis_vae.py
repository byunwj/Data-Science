from tensorflow.keras import layers

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python import training


class Sampling(layers.Layer):
    

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    

    def __init__(self, latent_dim, intermediate_dim=128):
        super(Encoder, self).__init__()
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
   
    def __init__(self, original_dim, intermediate_dim=128):
        super(Decoder, self).__init__()
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_output = layers.Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
    

    def __init__(self, original_dim, latent_dim, intermediate_dim=128):
        super(VariationalAutoEncoder, self).__init__()
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(original_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed
    
    def model(self):
        x = Input( shape = (self.original_dim, ))
        output = self.call(x)
        return Model(inputs = x, outputs = output, name = "variational autoencoder")


vae= VariationalAutoEncoder(15, 3)
vae.model().summary()
vae.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
vae(tf.ones((1, 15)))
vae.load_weights("./training/Epoch_002_Val_22.741.hdf5")
print( vae.encoder( tf.ones((1,15)) ) )

#autoencoder = vae.model()
#autoencoder.load_weights("./training/Epoch_002_Val_22.741.hdf5")



from numpy.core.numeric import indices
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

from tensorflow.python.ops.gen_batch_ops import batch



def get_model(loss_fn, opt):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(20, activation = 'relu', input_dim = 784))
    model.add(tf.keras.layers.Dense(1, activation = 'relu'))
    model.compile(optimizer = opt, loss = loss_fn)
    return model
    
#@tf.function
def train(model, loss_fn, opt, x_train, y_train, epochs, batch_size):
    for i in range(epochs):
        loss_lst = []
        #for i in range(int(x_train.shape[0]/batch_size)):
        #    indices = random.sample(range(x_train.shape[0]), batch_size)
        #    indices = random.shuffle(indices)
        #    x_batch = x_train[indices]
        #    y_batch = y_train[indices]
        for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size):
            with tf.GradientTape() as tape:
                y = model(x_batch, training=True)
                loss = loss_fn(y_batch, y)
                loss_lst.append(loss.numpy())
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
        print(np.mean(loss))
    
    return loss_lst


if __name__ == '__main__':
    # Load example MNIST data and pre-process it
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    
    loss_fn = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam() 

    model = get_model(loss_fn, opt)
    
    """
    _ = model.fit(x_train, y_train,
            batch_size=64,
            epochs=5,
            verbose=1,
            validation_data= (x_test, y_test)
            )
    """
    
    loss_lst = train(model, loss_fn, opt, x_train, y_train, 5, 64)
    
from numpy.core.numeric import indices
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import random

from tensorflow.python.ops.gen_batch_ops import batch

import timeit


# for categorical cross entropy : the y labels must be one hot vector
# for sparse categorical cross entropy : the y labels must be class-indices e.g. 0,1,2,5,6 
#                                        and the output of the model can still be (10,) with softmax



def get_model(loss_fn, opt):
    inputs = tf.keras.Input(shape=(784,), name='digits')
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer = opt,
                  loss = loss_fn,
                  metrics=['sparse_categorical_accuracy'])
    model.summary()

    return model
    

@tf.function
def update_weights(x_batch, y_batch, model, loss_fn, opt):
    with tf.GradientTape() as tape:
        y = model(x_batch, training=True)
        loss = loss_fn(y_batch, y)
        accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_batch, y)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return loss, accuracy


def train(model, loss_fn, opt, x_train, y_train, epochs, batch_size, train_dataset):
    #for i in range(epochs):
    #    for x_batch, y_batch in train_dataset:
    #        loss, accuracy = update_weights(x_batch, y_batch, model, loss_fn, opt)
    #    print('epoch {} \t loss:'.format(i+1), np.mean(loss), '/ accuracy:', np.mean(accuracy))

    for i in range(epochs):
        for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size):
            loss, accuracy = update_weights(x_batch, y_batch, model, loss_fn, opt)
        print('epoch {} \t loss:'.format(i+1), np.mean(loss), '/ accuracy:', np.mean(accuracy))


if __name__ == '__main__':
    # Load example MNIST data and pre-process it
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    opt = tf.keras.optimizers.Adam() 

    model = get_model(loss_fn, opt)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(64)

    #model.fit(train_dataset, epochs = 5)
    
    """
    print('time:', timeit.timeit(lambda: model.fit(x_train, y_train,
            batch_size=64,
            epochs=5,
            verbose=1,
            validation_data= (x_test, y_test)
            ) , number = 5))    
    """
    
    
    
    print('time:', timeit.timeit(lambda: train(model, loss_fn, opt, x_train, y_train, 5, 64, train_dataset), number = 5))
    
    
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss


class FeatureLoss(Loss):
    """
    able to see/minimizze the mean absolute error (could be another loss) for a specific feature when multiple feature outputs are predicted
    """
    
    def __init__(self, feature):
        super().__init__()
        self.feature = feature
    
    def call(self, y_true, y_pred):
        error = K.mean(K.abs(y_pred - y_true)[:, self.feature])
        return error


class HuberLoss(Loss):

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    # the call function that gets executed when an object is instantiated from the class
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self.threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = self.threshold * (tf.abs(error) - (0.5 * self.threshold))
        return tf.where(is_small_error, small_error_loss, big_error_loss)


class ContrastiveLoss(Loss):

    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin
    def call(self, y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(self.margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)



# inputs
#xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
xs = np.arange(30).reshape(6,5).astype(np.float)

# labels
#ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
ys = np.arange(30).reshape(6,5).astype(np.float)

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[5,])])
model.compile(optimizer='sgd', loss= FeatureLoss(2))
model.fit(xs, ys, epochs=500, verbose=2)






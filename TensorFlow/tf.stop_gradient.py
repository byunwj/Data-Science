import tensorflow as tf
import numpy as np
from tensorflow.python.ops.numpy_ops.np_array_ops import var

tf.compat.v1.disable_eager_execution()

'''
to see the effect of using tf.stop_gradient: (1). pass in 'stop_gradient' for freeze_method parameter when initializing TFExample class object
                                             (2). run this python file

to see the effect of using var_list paramter: (1). pass in 'var_list' for freeze_method parameter when initializing TFExample class object
                                              (2). run this python file
'''

class TFExample():
    def __init__(self, input_dim, hidden_dim, output_dim, freeze_method = None):
        self.sess = tf.compat.v1.Session()
        self.freeze_method = freeze_method

        self.x = tf.compat.v1.placeholder(tf.float32,[None, input_dim])
        self.y = tf.compat.v1.placeholder(tf.float32,[None, output_dim])

        with tf.compat.v1.variable_scope('not_update'):
            self.w1 = tf.compat.v1.Variable( tf.compat.v1.truncated_normal( shape = [input_dim, hidden_dim]) )
        with tf.compat.v1.variable_scope('update'):
            self.w2 = tf.compat.v1.Variable( tf.compat.v1.truncated_normal( shape = [hidden_dim, output_dim]) )
        
        self.output = self.make_output(self.x)
        self.loss = self.output - self.y

        if self.freeze_method == 'var_list':
            self.update_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='update')
            self.optimizer = tf.compat.v1.train.AdamOptimizer(0.01).minimize(self.loss, var_list = self.update_vars)
        else:
            self.optimizer = tf.compat.v1.train.AdamOptimizer(0.01).minimize(self.loss)


        self.train_x = np.random.normal(size = (4, input_dim))
        self.train_y = np.random.normal(size = (4, output_dim))
        
        self.sess.run(tf.compat.v1.global_variables_initializer())


    def make_output(self, x):
        if self.freeze_method == 'stop_gradient':
            hidden = tf.stop_gradient(tf.matmul(x,self.w1))
        else:
            hidden = tf.matmul(x, self.w1)

        output = tf.matmul(hidden, self.w2)

        return output

    def train(self, epochs = 1, data = None):
        for epoch in range(epochs):
            self.sess.run([self.optimizer], 
                           feed_dict = {self.x: self.train_x, self.y: self.train_y})



if __name__ == '__main__':
    np.random.seed(131)
    tfe = TFExample(2, 4, 1)

    w1_val, w2_val = tfe.sess.run([tfe.w1, tfe.w2])
    print("\n initial weights:")
    print("\n w1 = \n",w1_val,"\n","\n w2 = \n",w2_val) 

    tfe.train(epochs= 50)
    w1_val, w2_val = tfe.sess.run([tfe.w1, tfe.w2])
    print("\n trained weights:")
    print("\n w1 = \n",w1_val,"\n","\n w2 = \n",w2_val)



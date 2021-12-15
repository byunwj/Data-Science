import tensorflow as tf
import numpy as np

"""
there are two ways to freeze certain weights:
1. use tf.stop_gradient()
2. specify only the weights that you want to be updated in var_list inside optim.minimize()ss
"""

tf.compat.v1.disable_eager_execution()

sess = tf.compat.v1.Session()

x = tf.compat.v1.placeholder(tf.float32,[3,2])
y = tf.compat.v1.placeholder(tf.float32,[3,4])
w1 = tf.compat.v1.Variable(tf.ones([2,3]))
w2 = tf.compat.v1.Variable(tf.ones([3,4]))

hidden = tf.stop_gradient(tf.matmul(x,w1))
#hidden = tf.matmul(x,w1)

output = tf.matmul(hidden,w2)
loss = output - y

optimizer = tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss)
#optimizer = tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss, var_list = w2)

sess.run(tf.compat.v1.global_variables_initializer())
w1_val, w2_val = sess.run([w1, w2])
print("\n initial weights:")
print("\n w1 = \n",w1_val,"\n","\n w2 = \n",w2_val)

sess.run([optimizer], feed_dict = {x:np.random.normal(size = (3,2)),y:np.random.normal(size = (3,4))})
w1_val, w2_val = sess.run([w1, w2])
print("\n changed weights:")
print("\n w1 = \n",w1_val,"\n","\n w2 = \n",w2_val)

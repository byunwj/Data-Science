Data Science
======================

# Unsupervised Representation Learning
I used a publicly available health dataset (심평원 건강검진 데이터) to see if well known unsupervised clustering techniques could be used to learn the latent features of the data. The three main methods used are: AutoEncoder (AE), Variational AutoEncoder (VAE), and t-SNE. AutoEncoder and Variational Autoencoder were used to see if there were apparent advantages of VAE's stochasticity in learning the latent features, and t-SNE was used as a benchmark classical ML algorithm as it is a well-known non-linear ML algorithm used for dimension reduction. It should be noted that when VAE was first invented it was introduced as a generative model within the Bayesian framework. However, when VAEs are discussed in the deep learning framework, they are often introduced as another "version" of AE, which is an oversimplification as AE's main purpose is manifold learning. 

## Method
The dataset contains anonymous health screening results containing 34 features. The main purpose was to see when the dimension was reduced (to 3d or 2d) if people in the same "health category" would be clustered together. This was based on the assumption that people in the same "heatlh category" would be clustered together in the high (original) dimensional space in the first place. In order to acheive the above mentioned purpose, the "health categories" had to be defined. I used total cholesterol level to define the categories. The categories were defined as following: high cholesterol group : > 280 mg/dL, normal cholesterol group : 175 mg/dL < chl < 178 mg/dL, and low cholesterol group : < 125 mg/dL. There are other ways to define cholesterol groups such as using LDL cholesterol level as well; however, I tried to keep it simple. Also, I have to admit that how the three groups are defined has a noticeable effect on how clearly the clusters are formed when visualized which implies some uncertainty. 
Each person was then given a label (which of course was not used in training) accordingly.   

13 features were selected that I thought could represent one's overall health, including BMI, waist size, blood pressures, etc. The data was normalized using scikit-learn's Normalizer (l2 norm). The dataset was then divided into training and validation (or you can call it test set in this case) sets. The training set was used for training the neural network, and after training, the validation set was fed into the encoder of AE or VAE which then compressed the validation data into 3d latent vectors. The same training/validation process was applied to t-SNE as well to capture t-SNE's overall performance on representation learning compared to AE/VAE. The 3d and 2d vectors were then plotted, and to see if the data points in the same cholesterol group are clustered together in the compressed dimensions, different colors were assigned to the data points in different groups. 


## Result Visualization
Below are the clustering results of t-SNE, AE, and VAE, respectively. It can be seen that the result of t-SNE forms very rough clusters with many of the data points being considerably far from the center of their clusters. In AE's result, more pronounced clusters are formed with a handful of data points seemingly being closer to another cluster's center. In VAE's result, similar result can be observed but with seemingly less data points being closer to another cluster's center. In order to preicsely see which model yielded more well-formed clusters, I did the following two:  
(1). calculated the silhouette score, which is a popular clustering metric, for the 3d latent vector result of each model
(2). further compressed the 3d latent vectors to 2d latent vectors using to see how easily information in the 3d latent vectors could be translated into 2d latent vectors and calculated the silhouette score again  

Since the VAE model includes the KL error in addition to the reconstruction error while the AE model includes just the reconstruction error, I thought it would be unfair to just compare the validation loss. Thus, for fairness, I used the saved weights of each model around the same epoch. Comparing the silhouette score of the 3d latent vectors, the result of AE model was scored 0.23 while the result of VAE model was scored 0.18. Comparing the silhouette score of the 2d latent vectors, 
the result of the result of AE model was scored 0.43 while the result of VAE model was scored 0.38. We can see that the silhouette scores for both 2d and 3d were higher for the AE model. However, the noteworthy part was that even though the 2d latent vector result of the VAE model formed much clearer clusters than that of the AE model when visualized, the silhouette score for the AE model was still higher. It was unexpected to see that nicer clusters when visualized do not necessarily translate to higher clustering metric.

For purely experimental purposes, I also applied K-means clustering algorithm to the 3d latent vector to see how a classical ML algorithm would define the clusters and calculated the silhouette score as if the cluster label given by K-means is the true cluster label. The resulting silhouette score was 0.43 which is significantly higher than that of AE and VAE results. It suggests that although the 3d latent vectors (from AE and VAE) seem to form pronounced clusters to some degree when visualized, it is hard to say the clusters are well formed metric-wise. Imagine if we did not know the true cluster labels, it would be extremely hard to define the clusters based on the 3d latent vectors.

    
<figure>
<img src="https://github.com/byunwj/Data-Science/blob/main/Unsupervised%20Clustering/3d_clustering_tsne.png?raw=true" width="550px" height="350px" title="px(픽셀) 크기 설정" alt="3D Clustering by t-SNE">
<figcaption>3D Clustering by t-SNE</figcaption>
</figure>
</br>

<figure>
<img src="https://github.com/byunwj/Data-Science/blob/main/Unsupervised%20Clustering/3d_clustering_ae2.png?raw=true" width="550px" height="350px" title="px(픽셀) 크기 설정" alt="3D Clustering by AutoEncoder">
<figcaption>3D Clustering by AutoEncoder</figcaption>
</figure>
</br>

<figure>
<img src="https://github.com/byunwj/Data-Science/blob/main/Unsupervised%20Clustering/3d_clustering_vae.png?raw=true" width="550px" height="350px" title="px(픽셀) 크기 설정" alt="3D Clustering by Variational AutoEncoder">
<figcaption>3D Clustering by Variational AutoEncoder</figcaption>
</figure>
</br>

# TensorFlow
## tf.stop_gradient() 

tf.stop_gradient() can be used when you do not want to update certain weights during backpropagation as below:  

```
x = tf.compat.v1.placeholder(tf.float32,[None, input_dim])
y = tf.compat.v1.placeholder(tf.float32,[None, output_dim])
w1 = tf.compat.v1.Variable( tf.compat.v1.truncated_normal( shape = [input_dim, hidden_dim]) )
w2 = tf.compat.v1.Variable( tf.compat.v1.truncated_normal( shape = [hidden_dim, output_dim]) )
hidden = tf.stop_gradient(tf.matmul(x, w1))
output = tf.matmul(hidden, w2)
loss = output - y
optimizer = tf.compat.v1.train.AdamOptimizer(0.01).minimize(loss)
```
As a result, after the weight updates, w1 remains the same while w2 gets updated as shown below:  

```
 initial weights:
 w1 = 
 [[ 1.1496285  -1.5963056  -1.4022949  -1.7677178 ]
 [ 0.45795143 -0.50486845  0.43018925 -1.2649945 ]] 
 
 w2 = 
 [[-0.37988296]
 [-0.35934356]
 [-1.4428393 ]
 [-1.4217583 ]]

 trained weights:
 w1 = 
 [[ 1.1496285  -1.5963056  -1.4022949  -1.7677178 ]
 [ 0.45795143 -0.50486845  0.43018925 -1.2649945 ]] 
 
 w2 = 
 [[-0.87988234]
 [ 0.14065644]
 [-1.9428387 ]
 [-0.9217589 ]]
```

There also seems to be another way to update only certain weights.  
It is by using the *var_list* parameter of the optimizer's minimize function.
The code snippet below would yield the same effect as using tf.stop_gradient() in the above code snippet.
Instead of creating variables under specifc scopes and retrieving them using get_collection like below, one can simply just pass in a list of the variables that should be updated. ( ex) var_list = [w2] )  
Creating variables under specific scopes becomes useful when the number of variables gets big. 

```
x = tf.compat.v1.placeholder(tf.float32,[None, input_dim])
y = tf.compat.v1.placeholder(tf.float32,[None, output_dim])
with tf.compat.v1.variable_scope('not_update'):
    w1 = tf.compat.v1.Variable( tf.compat.v1.truncated_normal( shape = [input_dim, hidden_dim]) )
with tf.compat.v1.variable_scope('update'):
    w2 = tf.compat.v1.Variable( tf.compat.v1.truncated_normal( shape = [hidden_dim, output_dim]) )
update_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='update')
hidden = tf.matmul(x, w1)      # not using tf.stop_gradient()
output = tf.matmul(hidden, w2)
loss = output - y
optimizer = tf.compat.v1.train.AdamOptimizer(0.01).minimize(loss, var_list = update_vars) # specify that the variables created under 
                                                                                          # 'update' scope should only be updated
```

Above are just example snippets, and more details can be found in tf.stop_gradient.py in TensorFlow folder.



## tf.GradientTape() & @tf.function decorator
As TF2 was developed aiming for tight integration with Keras and user friendliness, model.fit() method is widely used for its simplicity. 
However, model.fit() lacks flexibility as the training step function is alreay defined inside the fit() function. Thus, tf.GradientTape() becomes useful when you want to write a custom training loop since you can override the training step function of the tf.keras.Model class.  
The code in tf.GradientTape.py in TensorFlow folder is to just show how to use tf.GradientTape() as opposed to model.fit(). The code snippet below illustrates using tf.GradientTape() to do the same job as model.fit().  

```
@tf.function
def update_weights(x_batch, y_batch, x_batch_val, y_batch_val, model, loss_fn, opt):
    with tf.GradientTape() as tape:
        y = model(x_batch, training=True)
        loss = loss_fn(y_batch, y)
        accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_batch, y)

        y_val = model(x_batch_val, training = False)
        val_loss = loss_fn(y_batch_val, y_val)
        val_accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_batch_val, y_val)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return loss, accuracy, val_loss, val_accuracy


def train(model, loss_fn, opt, epochs, train_dataset, val_dataset):
    for i in range(epochs):
        for (x_batch, y_batch), (x_batch_val, y_batch_val) in zip(train_dataset, val_dataset) :
            loss, accuracy, val_loss, val_accuracy = update_weights(x_batch, y_batch, x_batch_val, y_batch_val, model, loss_fn, opt)
        print('epoch {} \t loss:'.format(i+1), np.mean(loss), '/ accuracy:', np.mean(accuracy), 
                                                              '\t val loss:', np.mean(val_loss),
                                                               '/ val accuray:', np.mean(val_accuracy), '\n')
```
It is also noteworthy to mention that applying @tf.function decorator to update_weights() function is crucial for training speed.
@tf.function converts a Python function to its graph representation. 



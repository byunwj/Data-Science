Data Science
======================

# Unsupervised Representation Learning
I used a publicly available health dataset (심평원 건강검진 데이터) to see if well known unsupervised clustering techniques could be used to learn the latent features. The three main methods used are: AutoEncoder (AE), Variational AutoEncoder (VAE), and t-SNE. AutoEncoder and Variational Autoencoder were used to see if there were apparent advantages of VAE's stochasticity in learning the latent features, and t-SNE was used as a benchmark classical ML algorithm as it is a well-known non-linear ML algorithm used for dimension reduction. It should be noted that when VAE was first invented it was introduced as a generative model within the Bayesian framework. However, when VAEs are discussed in the deep learning framework, they are often introduced as a "version" of AE, which is an oversimplification as AE's main purpose is manifold learning. 

## Method
The dataset contains anonymous health screening results containing 34 features. The main purpose was to see when the dimension was reduced (to 3d or 2d) if people in the same "health category" would be clustered together. This was based on the assumption that people in the same "heatlh category" would be clustered together in the high (original) dimensional space in the first place. In order to acheive the above mentioned purpose, the "health categories" had to be defined. I used total cholesterol level to define the categories. The categories were defined as following: high cholesterol group : > 280 mg/dL, normal cholesterol group : 175 mg/dL < chl < 178 mg/dL, and low cholesterol group : < 125 mg/dL. There are other ways to define cholesterol groups such as using LDL cholesterol level as well; however, I tried to keep it simple. Also, I have to admit that how the three groups are defined has a noticeable effect on how clearly the clusters are formed when visualized which implies some uncertainty. 
Each person was then given a label (which of course was not used in training) accordingly.   

13 features were selected that I thought could represent one's overall health, including BMI, waist size, blood pressures, etc. The data was normalized using scikit-learn's Normalizer (l2 norm). The dataset was divided into training, validation sets. The training set was used for training the neural network, and after training, the validation set was fed into the encoder of AE or VAE which then compressed the validation data into 3d latent vectors. For purely experimental purposes, t-SNE was then applied to map the 3d latent vectors to 2d which is also easier on the eye when visualized. The same training/validation process was applied to t-SNE as well to capture t-SNE's overall performance on representation learning compared to AE/VAE. The 3d and 2d vectors were then plotted, and to see if the data points in the same cholesterol group are clustered together in the compressed dimension, different colors were assigned to the data points in different groups. 


## Result Visualization
Below are the clustering results of t-SNE, AE, and VAE, respectively. It can be seen that the result of t-SNE forms very rough clusters with many of the data points being considerably far from the center of their clusters. In AE's result, more pronounced clusters are formed with a handful of data points being closer to another cluster's center. In VAE's result, similar result can be observed but with seemingly less data points being mislaid. In order to preicsely see which model yielded a clearer clusters, K-means clustering algorithm will be applied to the latent vectors and compare the result (cluster label) with the original group label.  
    
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
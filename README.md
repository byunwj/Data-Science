Data Science
======================

# Unsupervised Clustering
I used a publicly available health dataset (심평원 건강검진 데이터) to see if well known unsupervised clustering techniques could be used to learn the latent features. The three main methods used are: AutoEncoder (AE), Variational AutoEncoder (VAE), and t-SNE. AutoEncoder and Variational Autoencoder were used to see if there were apparent advantages of VAE's stochasticity in learning the latent features, and t-SNE was used as a benchmark classical ML algorithm as it is a well-known non-linear ML algorithm used for dimension reduction. It should be noted that when VAE was first invented it was introduced as a generative model within the Bayesian framework. However, when VAEs are discussed in the deep learning framework, they are often introduced as a "version" of AE, which is an oversimplification as AE's main purpose is manifold learning.  

## Method
The dataset contains anonymous health screening results containing 34 features. The main purpose was to see when the dimension was reduced (to 3d or 2d) if people in the same "health category" would be clustered together. This was based on the assumption that people in the same "heatlh category" would be clustered together in the high (original) dimensional space in the first place. In order to acheive the mentioned purpose, the "health categories" had to be defined. 
13 features were selected that I thought could represent one's overall health, including BMI, waist size, blood pressures, etc. 
The data was normalized using scikit-learn's Normalizer (l2 norm). 


## Result Visualization

Below are the clustering results of the three methods are shown. The first graph represents the 3D clustering result using the t-SNE algorithm. 

    
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
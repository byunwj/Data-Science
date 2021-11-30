Data Science
======================

# Unsupervised Clustering
I used a publicly available health dataset (심평원 건강검진 데이터) to see if well known unsupervised clustering techniques could be used to learn the latent features. The three main methods used are: AutoEncoder (AE), Variational AutoEncoder (VAE), and t-SNE. AutoEncoder and Variational Autoencoder were used to see if there were apparent advantages of VAE's stochasticity in learning the latent features, and t-SNE was used as a benchmark classical ML algorithm. It should be noted that when VAE was first invented it was introduced as a generative model within the Bayesian framework. However, when VAEs are discussed in the deep learning framework, they are often introduced as a "version" of AE, which is an oversimplification as AE's main purpose is manifold learning.

<img src="https://github.com/byunwj/Data-Science/blob/main/Unsupervised%20Clustering/3d_clustering_ae.png?raw=true" width="450px" height="300px" title="px(픽셀) 크기 설정" alt="3D Clustering by AutoEncoder"></img><br/>



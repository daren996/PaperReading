## PCA & Autoencoder & K-Means



## Term

- In **dimensionality reduction**, we try to learn a mapping to a lower dimensional space that preserves as much information as possible about the input. 
- The **projection** of a point x onto S is the point   ̃x ∈ S closest to x. In machine learning,  ̃x is also called the **reconstruction** of x. z is its **representation**, or code. z = U^⊤(x − μ) 
- An **autoencoder** is a feed-forward neural net whose job it is to take an input x and predict x.



## Dimensionality Reduction

In dimensionality reduction, we try to learn a mapping to a lower dimensional space that preserves as much information as possible about the input. 

Motivations: Save computation/memory; Reduce overfitting; Visualize in 2 dimensions.

Dimensionality reduction can be linear or nonlinear. 

### Linear Dimensionality Reduction 

Suppose there are points in the three-dimensional space. These points are distributed on an inclined plane that passes through the origin. 

<img src='https://github.com/daren996/PaperReading/blob/master/ML/Img/PCS-PLANE.png' width=80% align=center >

#### Natural Coordinate System

If you use the three axes of the natural coordinate system x, y, and z to represent the set of data, you need to use three dimensions. However, the distribution of these points is only on a two-dimensional plane. 

#### Rotated Coordinate System

If the rotated coordinate system is denoted as x', y', z', then the representation of the set of data can only be represented by the two dimensions x' and y'. To recover the original representation, we can save the transformation matrix between these two coordinates. 

#### Linear Dependent

If the data is arranged in a matrix by row, then the rank of this matrix is 2. There is a correlation between these data, and the largest linearly independent group of vectors over the origin of these data contains 2 vectors, if the plane is assumed to pass the origin.  

#### Data Centering

What if the plane does not pass the origin? Data centering! Translate the origin of the coordinates to the data center so that the originally unrelated data is relevant in this new coordinate system. Interestingly, as three points must be coplanar, any three points after cemtering in the three-dimensional space are linearly related. 

#### Projection

Assuming that the data has a small noise on the z' axis, then we still use the above two-dimensional representation of the data, because we believe that the information of the two axes is the principal component of the data, and this information is enough for our further analysis.

At the same time, if there is noise or redundancy in the features, we also need a feature dimension reduction method to reduce the number of features, reduce noise and redundancy, the possibility of overfitting. 

## PCA

Choosing a subspace to maximize the projected variance, or minimize the reconstruction error, is called **principal component analysis (PCA)**.

Σ is the empirical covariance matrix. Covariance matrices are symmetric and positive semidefinite.

The optimal PCA subspace is spanned by the top K eigenvectors of Σ. More precisely, choose the first K of any orthonormal eigenbasis for Σ.

These eigenvectors are called **principal components**, analogous to the principal axes of an ellipse.



Dimensionality reduction aims to find a low-dimensional representation of the data.

PCA projects the data onto a subspace which maximizes the projected variance, or equivalently, minimizes the reconstruction error.

The optimal subspace is given by the top eigenvectors of the empirical covariance matrix.

PCA gives a set of decorrelated features.



## Autoencoder

Why autoencoders?

1. Map high-dimensional data to two dimensions for visualization.

2. Learn abstract features in an unsupervised way so you can apply them to a supervised task. 
   1. Unlabled data can be much more plentiful than labeled data

For Linear Autoencoders: 

​	The optimal weights for a linear autoencoder are just the principal components.

Deep nonlinear autoencoders learn to project the data, not onto a subspace, but onto a nonlinear manifold. Nonlinear autoencoders can learn more powerful codes for a given dimensionality, compared with linear autoencoders.



## K-Means

Grouping data points into clusters, with no labels, is called **clustering**.

### Process

Initialization: randomly initialize cluster centers, m_1 , . . . , m_K

Repeat until convergence (until assignments do not change):

1. Assignment step: Assign each data point to the closest cluster, k_n = argmin d(m_k, x^n)
2. Refitting step: Move each cluster center to the center of gravity of the data assigned to it

### Why K-means Converges

Whenever an assignment is changed, the sum squared distances J of data points from their assigned cluster centers is reduced.

Whenever a cluster center is moved, J is reduced.

Test for convergence: If the assignments do not change in the assignment step, we have converged (to at least a local minimum).

Test for convergence: If the assignments do not change in the assignment step, we have converged (to at least a local minimum).


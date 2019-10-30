# MachineLearning

Some resources about machine learning.

## Structure

Structure of this folder.

3. [CSC2515](http://www.cs.toronto.edu/~rgrosse/courses/csc2515_2019/): A Course I took part in at the University of Toronto.
2. [Information Theory](https://homes.cs.washington.edu/~anuprao/pubs/CSE533Autumn2010/): Useful slides about Information Theory from the Univ. of Washington.  
3. Neural Networks. 

Papers in this folder:

1. AlexNet.pdf
2. Occam’s Razor.pdf



## Scikit Learn

[Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

[Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html): Convert a collection of text documents to a matrix of token counts. 

Model Selection. 
1. [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split): split arrays or matrices into random train and test subsets. 
2. [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score): evaluate a score by cross-validation. 







## Terms

**Generalization Error**: Error rate on new examples. We would like our algorithm to **generalize** to data it hasn’t before. (referring lec01-slides)

**Regularization**: The process of adding information in order to solve an ill-posed problem or to prevent overfitting. **Regularizer**: a function that quantifies how much we prefer one hypothesis vs. another. We can improve the generalization by adding a regularizer. 

**Bayes optimal classifier**: The function f(x) which minimizes the misclassification rate. Its error rate is called the **Bayes error**. **Bayes consistency** is a very special property, and holds for hardly any of the algorithms covered in this course.

**Bias**: how wrong the expected prediction is (corresponds to underfitting). 

**Variance**: the amount of variability in the predictions (corresponds to overfitting).  

**Overfitting** is a modeling error which occurs when a function is too closely fit to a limited set of data points. **Overfitting** the model generally takes the form of making an overly complex model to explain idiosyncrasies in the data under study. 

**Underfitting** occurs when a statistical model or machine learning algorithm cannot capture the underlying trend of the data. Intuitively, **underfitting** occurs when the model or the algorithm does not fit the data well enough. Specifically, **underfitting** occurs if the model or algorithm shows low variance but high bias.

An **ensemble** of predictors is a set of predictors whose individual decisions are combined in some way to classify new examples. 

**Bagging**: Train classifiers independently on random subsets of the training data. Reduce the variance but have no effect of the bias. **Bootstrap aggregation**: sample new small datasets on a simple dataset.

**Boosting**: Train classifiers sequentially, each time focusing on training examples that the previous ones got wrong. 

If the training examples can be separated by a linear decision rule, they are **linearly separable**.

**Stochastic Gradient Descent**: update the parameters based on the gradient for a single training example. **Mini-batch**: Compute the gradients on a medium-sized set of training examples. Each entire pass over the dataset is called an **epoch**. 



## KNN

#### 1. Tradeoffs in choosing k? 

Small k 

- Good at capturing **fine-grained** patterns 
- May **overfit**, i.e. be sensitive to random idiosyncrasies in the training data 

Large k 

- Makes **stable** predictions by averaging over lots of examples 
- May **underfit**, i.e. fail to capture important regularities 

Rule of thumb: k < sqrt(n), where n is the number of training examples 

k influnces the underfitting and overfitting, as we can’t fit as part of the learning algorithm itself. We can tune **hyperparameters** using a validation set. 

#### 2. Is KNN bayes consistent?

In other words, given enough data, will it give the “right” answer? 

The **Bayes optimal classifier** is the function f(x) which minimizes the misclassification rate. Its error rate is called the **Bayes error**.

The asymptotic error of 1-NN is at most twice the Bayes error. 

The KNN approaches the Bayes error, i.e. KNN is Bayes consistent. (Central Limit Theorem)

#### 3. Curse of Dimensionality

KNN suffers from the Curse of Dimensionality. 

One perspective: When d is larger, larger **distance** we will get with the increase of the fraction of volume. And we need more balls to cover the volume. (How large does N need to be to guarantee we have an ε-neighbour?)

Another perspective: In high dimensions, “most” points are approximately the same **distance**. 

Probable Solution: project to get **intrinsic dimension**.

#### 4. Normalization 

Nearest neighbors can be sensitive to the ranges (**units**) of different features. 

**Normalize** each dimension to be zero mean and unit variance. 

In some cases, the scale might be important.

#### 5. Complexity

Number of computations at training time: 0

Number of computations at test time, per query (na ̈ıve algorithm): 

- Calculuate D-dimensional Euclidean distances with N data points: O(ND) 
- Sort the distances: O(NlogN)

Need to store the entire dataset in memory.

## Decision Trees

Learning the simplest (smallest) decision tree is an NP complete problem.

Greedy heuristic:

- Start from an empty decision tree 
- Split on the “**best**” attribute (choose attribute that gives the highest gain)
- Recurse 

####  1. Division Methods

Measurement of Uncertainty: 

1. **Entropy**. <img src="http://latex.codecogs.com/gif.latex?H(X)=-\sum_{x\in X}{p(x)log_2p(x)}" />

2. **Conditional Entropy.** <img src="http://latex.codecogs.com/gif.latex?H(Y|X)=\sum_{x\in X}{p(x)H(Y|X=x)}" />

Three common division methods of classification decision tree (algorithms used to develop decision trees):

1. ID3, **Information Gain**. <img src="http://latex.codecogs.com/gif.latex?IG(D|A)=H(D)-H(D|A)" />

2. C4.5, Ratio of Information Gain. <img src="http://latex.codecogs.com/gif.latex?g_R(D,A)=\frac{g(D,A)}{H_A(D)}" />

3. CART, Gini Coefficient. <img src="http://latex.codecogs.com/gif.latex?Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2)" />  


#### 2. Random Forests

Random forests = bagged decision trees, with one extra trick to decorrelate the predictions.

> When choosing each node of the decision tree, choose a random set of d input features, and only consider splits on those features
> Random forests are probably the best black-box machine learning algorithm — they often work well with no tuning whatsoever. One of the most widely used algorithms in Kaggle competitions.

#### 3.1. Advantages of decision trees over KNN

1. Good when there are lots of attributes, but only a few are important Good with discrete attributes
2. Easily deals with missing values (just treat as another value).
3. Robust to scale of inputs.
4. Fast at test time.
5. More interpretable.

#### 3.2. Advantages of KNN over decision trees

1. Few hyperparameters.
2. Able to handle attributes/features that interact in complex ways (e.g. pixels). 
3. Can incorporate interesting distance measures (e.g. shape contexts). 
4. Typically make better predictions in practice. Ensembles of decision trees are much stronger. But they lose many of the advantages listed above.

## Linear Regression/Classification

#### 1. Why vectorize? 

The equations, and the code, will be simpler and more readable. Gets rid of dummy variables/indices!

Vectorized code is much faster.

1. Cut down on Python interpreter overhead
2. Use highly optimized linear algebra libraries
3. Matrix multiplication is very fast on a Graphics Processing Unit (GPU) 

#### 2. Direct Solution

In python, 

```python
y = np.dot(X, w) + b
cost = np.sum((y - t) ** 2) / (2. * N)
```

Direct solution: <img src="http://latex.codecogs.com/gif.latex?\textbf{w}=(\textbf{X}^T\textbf{X}^{-1})\textbf{X}^T\textbf{t}" />

Linear regression is one of only a handful of models in this course that permit direct solution. 

Why **gradient descent**, if we can find the optimum directly? 

1. GD can be applied to a much broader set of models 
2. GD can be easier to implement than direct solutions, especially with automatic differentiation software
3. For regression in high-dimensional spaces, GD is more efficient than direct solution (matrix inversion is an O(D3) algorithm). 

#### 3. Polynomial Regression 

The **degree** of the polynomial is a hyperparameter, just like k in KNN. We can tune it using a validation set. 

Instead of restricting the **degree** of the model, we could use another approach: **regularization**. Regularizer: a function that quantifies how much we prefer one hypothesis vs. another. Methods: Try to keep the coefficients small. 

#### 4. Logistic Regression 

**Problem** of 0-1 regression: protial derivitive is 0.

**Problem** of **surrogate loss function** (least squares): The loss function hates when you make correct predictions with high confidence.

Logistic with least squares:

- The logistic function is a kind of **sigmoidal**, or S-shaped, function.
- A linear model with a logistic nonlinearity is known as **log-linear**.
- Used in this way, σ is called an activation function, and z is called the **logit**. 

**Problem**: the loss function saturates. 

**Cross-entropy loss**: <img src="http://latex.codecogs.com/gif.latex?\textbf{L}(y,t)=-t\log(y)-(1-t)\log(1-y)" />

#### 5. SoftMax Function

Softmax function: <img src="http://latex.codecogs.com/gif.latex?softmax(z_1, ..., z_K)_k=\frac{e^{Z_k}}{\sum_{k'}{e^{Z_{k'}}}}" />

Outputs are positive and sum to 1 (so they can be interpreted as probabilities).

**Softmax-cross-entropy**:  Softmax regression.

#### 6. L1 vs. L2 Regularization 

For L1, 

- In general, the optimal weight vector will be **sparse**, i.e. many of the weights will be exactly zero. This is useful in situations where you have lots of features, but only a small fraction of them are likely to be relevant (e.g. genetics). 
- The cost function is a quadratic program, a more difficult optimization problem than for L2 regularization. 

## SVM

**Optimal Separating Hyperplane**: A hyperplane that separates two classes and maximizes the distance to the closest point from either class, i.e., maximize the **margin** of the classifier. 

The important training examples are the ones with algebraic margin 1, and are called **support vectors**.

Hence, this algorithm is called the (hard) **Support Vector Machine** (SVM) (or Support Vector Classifier).

SVM-like algorithms are often called **max-margin** or **large-margin**.

#### 1. Soft-margin SVM 

Allow some points to be within the margin or even be misclassified; we represent this with **slack variables**. But constrain or penalize the total amount of slack. 

Thelossfunction L_H(y,t)=(1−ty)_+ is called the **hinge loss**. 

The soft-margin SVM can be seen as a linear classifier with hinge loss and an L2 regularizer. 

## AdaBoost

Boosting: Train classifiers sequentially, each time focusing on training data points that were previously misclassified. 

Key steps of AdaBoost: 

1. At each iteration we re-weight the training samples by assigning larger weights to samples (i.e., data points) that were classified incorrectly. 
2. We train a new weak classifier based on the re-weighted samples. 
3. We add this weak classifier to the ensemble of classifiers. This is our new classifier. 
4. We repeat the process many times. 


```python
α = 1/2 log(1/err - 1)
w = w exp(2α)
```

## Neural Networks

All input units are connected to all output units. We call this a **fully connected layer**.

A multilayer network consisting of fully connected layers is called a **multilayer perceptron**. 

Multilayer feed-forward neural nets with nonlinear activation functions are **universal function approximators**: they can approximate any function arbitrarily well.

Limits of **universality**: 

1. You may need to represent an exponentially large network. 
2. If you can learn any function, you’ll just overfit.
3. Really, we desire a compact representation! 

#### 1. Backpropagation 

**Computation graph**: The nodes represent all the inputs and computed quantities, and the edges represent which nodes are computed directly as a function of which other nodes. 

**Multivariate Chain Rule** 

**Computational Cost**: the backward pass is about as expensive as two forward passes. 

#### 2. Gradient Checking

Check your derivatives numerically by plugging in a small value of h, This is known as **finite differences**.

Gradient checking is really important!

Learning algorithms often appear to work even if the math is wrong. But: 

1. They might work much better if the derivatives are correct. 
2. Wrong derivatives might lead you on a wild goose chase. 

#### 3. Convex

Unfortunately, training a network with hidden units cannot be convex because of permutation symmetries. We can re-order the hidden units in a way that preserves the function computed by the network. 

Because of permutation symmetry, there are K! permutations of the hidden units in a given layer which all compute the same function. 

Suppose we average the parameters for all K! permutations. Then we get a degenerate network where all the hidden units are identical. 

If the cost function were convex, this solution would have to be better than the original one, which is ridiculous! 

Hence, training multilayer neural nets is non-convex. 

#### 4. Convolution 

The thing we convolve by is called a **kernel**, or **filter**. 

What do these filters do? 

	# Blur
	0 1 0
	1 4 1
	0 1 0
	
	# Sharppen
	0 -1  0
	-1 8 -1
	0 -1  0
	
	# Edge
	0 -1  0
	-1 4 -1
	0 -1  0
	
	# Vertical Edge
	1 0 -1
	2 0 -2
	1 0 -1

**Hyperparameters** of a convolutional layer:

1. The number of filters (controls the **depth** of the output volume) 
2. The **stride**: how many units apart do we apply a filter spatially (this controls the spatial size of the output volume) 
3. The size w × h of the filters  

**Pooling**  gain robustness to the exact spatial location of features. 

#### 5. Ways to measure the size of a network

1. Number of units. This is important because the activations need to be stored in memory during training (i.e. backprop). 
2. Number of weights. This is important because the weights need to be stored in memory, and because the number of parameters determines the amount of overfitting. 
3. Number of connections. This is important because there are approximately 3 add-multiply operations per connection (1 for the forward pass, 2 for the backward pass). 

Most of the units and connections are in the convolution layers. 

Most of the weights are in the fully connected layers. 

#### 6. Visualization

we can understand what first-layer features are doing by visualizing the weight matrices. 

Visualize higher-level features by seeing what inputs activate them. 

One way to formalize: pick the images in the training set which activate a unit most strongly. 

Problems? 

1. Can’t tell what the unit is actually responding to in the image. 

2. We may read too much into the results, e.g. a unit may detect red, and the images that maximize its activation will all be stop signs.

Optimizing the Image 

- Optimize an image from scratch to increase a unit’s activation. 

- Can do gradient ascent on an image to maximize the activation of a given neuron. 

- Higher layers in the network often learn higher-level, more interpretable representations 




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
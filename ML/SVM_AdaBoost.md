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


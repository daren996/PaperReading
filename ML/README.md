# MachineLearning

Some resources about machine learning.

## Structure

Structure of this folder.

1. Decision Tree. 
2. Information Theory. 
3. CSC2515. A Course I took part in at the University of Toronto.



## Scikit Learn

[Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

[Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html): Convert a collection of text documents to a matrix of token counts. 

Model Selection. 
1. [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split): split arrays or matrices into random train and test subsets. 
2. [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score): evaluate a score by cross-validation. 



## Discription

[Course CSC2515](http://www.cs.toronto.edu/~rgrosse/courses/csc2515_2019/): A Course I took part in at the University of Toronto.

[Information Theory](https://homes.cs.washington.edu/~anuprao/pubs/CSE533Autumn2010/): Useful slides about Information Theory from the Univ. of Washington.  





## Terms

**Generalization Error**: Error rate on new examples. We would like our algorithm to **generalize** to data it hasn’t before. (referring lec01-slides)

**Regularization**: The process of adding information in order to solve an ill-posed problem or to prevent overfitting. 

**Bayes optimal classifier**: The function f(x) which minimizes the misclassification rate. Its error rate is called the **Bayes error**. **Bayes consistency** is a very special property, and holds for hardly any of the algorithms covered in this course.

**Overfitting** is a modeling error which occurs when a function is too closely fit to a limited set of data points. **Overfitting** the model generally takes the form of making an overly complex model to explain idiosyncrasies in the data under study. 

**Underfitting** occurs when a statistical model or machine learning algorithm cannot capture the underlying trend of the data. Intuitively, **underfitting** occurs when the model or the algorithm does not fit the data well enough. Specifically, **underfitting** occurs if the model or algorithm shows low variance but high bias.



## KNN

#### Tradeoffs in choosing k? 

Small k 

- Good at capturing **fine-grained** patterns 
- May **overfit**, i.e. be sensitive to random idiosyncrasies in the training data 

Large k 

- Makes **stable** predictions by averaging over lots of examples 
- May **underfit**, i.e. fail to capture important regularities 

Rule of thumb: k < sqrt(n), where n is the number of training examples 

k influnces the underfitting and overfitting, as we can’t fit as part of the learning algorithm itself. We can tune **hyperparameters** using a validation set. 

#### Is KNN bayes consistent?

In other words, given enough data, will it give the “right” answer? 

The **Bayes optimal classifier** is the function f(x) which minimizes the misclassification rate. Its error rate is called the **Bayes error**.

The asymptotic error of 1-NN is at most twice the Bayes error. 

The KNN approaches the Bayes error, i.e. KNN is Bayes consistent. (Central Limit Theorem)

#### Curse of Dimensionality

KNN suffers from the Curse of Dimensionality. 

One perspective: When d is larger, larger **distance** we will get with the increase of the fraction of volume. And we need more balls to cover the volume. 

Another perspective: In high dimensions, “most” points are approximately the same **distance**. 

Probable Solution: project to get **intrinsic dimension**.

#### Normalization 

Nearest neighbors can be sensitive to the ranges (**units**) of different features. 

**Normalize** each dimension to be zero mean and unit variance. 

In some cases, the scale might be important.

## Decision Trees

Learning the simplest (smallest) decision tree is an NP complete problem.

Greedy heuristic:

- Start from an empty decision tree 
- Split on the “**best**” attribute 􏰇 
- Recurse 

#### Division Methods

Three common division methods of classification decision tree (algorithms used to develop decision trees):

1. ID3, Information Gain. <img src="http://latex.codecogs.com/gif.latex?g(D,A)=H(D)-H(D|A)" />

2. C4.5, Ratio of Information Gain. <img src="http://latex.codecogs.com/gif.latex?g_R(D,A)=\frac{g(D,A)}{H_A(D)}" />

3. CART, Gini Coefficient. <img src="http://latex.codecogs.com/gif.latex?Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2)" />  


Measurement of Uncertainty: 

**Entropy**. <img src="http://latex.codecogs.com/gif.latex?H(X)=-\sum_{x\in X}{p(x)log_2p(x)}" />

**Conditional Entropy.** <img src="http://latex.codecogs.com/gif.latex?H(Y|X)=\sum_{x\in X}{p(x)H(Y|X=x)}" />





#### Random Forests

Random forests = bagged decision trees, with one extra trick to decorrelate the predictions.

> When choosing each node of the decision tree, choose a random set of d input features, and only consider splits on those features
> Random forests are probably the best black-box machine learning algorithm — they often work well with no tuning whatsoever. One of the most widely used algorithms in Kaggle competitions.

#### Advantages of decision trees over KNN

1. Good when there are lots of attributes, but only a few are important Good with discrete attributes
2. Easily deals with missing values (just treat as another value).
3. Robust to scale of inputs.
4. Fast at test time.
5. More interpretable.

#### Advantages of KNN over decision trees

1. Few hyperparameters.
2. Able to handle attributes/features that interact in complex ways (e.g. pixels). 
3. Can incorporate interesting distance measures (e.g. shape contexts). 
4. Typically make better predictions in practice. As we’ll see next lecture, ensembles of decision trees are much stronger. But they lose many of the advantages listed above.



## Complexity 

### KNN

Number of computations at training time: 0

Number of computations at test time, per query (na ̈ıve algorithm): 

- Calculuate D-dimensional Euclidean distances with N data points: O(ND) 
- Sort the distances: O(NlogN)

Need to store the entire dataset in memory.

### Decision Trees






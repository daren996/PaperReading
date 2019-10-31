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
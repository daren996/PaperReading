# Final



## Term

- In **dimensionality reduction**, we try to learn a mapping to a lower dimensional space that preserves as much information as possible about the input. 

- The **projection** of a point x onto S is the point   ̃x ∈ S closest to x. In machine learning,  ̃x is also called the **reconstruction** of x. z is its **representation**, or code. z = U^⊤(x − μ) 

- An **autoencoder** is a feed-forward neural net whose job it is to take an input x and predict x.

- Grouping data points into clusters, with no labels, is called **clustering**.

- **K-means** assumes there are k clusters, and each point is close to its cluster center (the mean of points in the cluster). **Initialization**: randomly initialize cluster centers. The algorithm iteratively alternates between two steps: **Assignment** step: Assign each data point to the closest cluster; **Refitting** step: Move each cluster center to the center of gravity of the data assigned to it. 

- The **likelihood function** is the probability of the observed data, as a function of θ. 

  最大似然估计的目的就是,利用已知样本结果,反推最有可能导致这样结果的参数值

  Good values of θ should assign high probability to the observed data. This motivates the maximum likelihood criterion.

- **Mahalanobis distance** (x − μ)^T Σ^−1 (x − μ) measures the distance from x to μ in a space stretched according to Σ. 可以应对高维线性分布的数据中各维度间非独立同分布的问题 ![img](https://pic4.zhimg.com/80/v2-d2987369d8167a362482d6cbecefb8bb_hd.jpg)

- Maximum likelihood has a pitfall: if you have too little data, it can overfit. Because it never observed T, it assigns this outcome probability 0. This problem is known as **data sparsity**.

- The **Bayesian parameter estimation** approach treats the parameters as random variables as well. 

  To define a Bayesian model, we need to specify two distributions:

  The **prior distribution** p(θ), which encodes our beliefs about the parameters before we observe the data. The **likelihood** p(D | θ), same as in maximum likelihood

  When we update our beliefs based on the observations, we compute the **posterior distribution** using Bayes’ Rule.

- Beta distribution as prior. 

  The prior and likelihood have the same functional form. This phenomenon is known as **conjugacy**, and it’s very useful.

- The **posterior predictive distribution** is the distribution over future observables given the past observations. We compute this by marginalizing out the parameter(s): p(D' | D) = ſ p(θ | D) p(D' | θ) dθ.

- **Maximum a-posteriori (MAP) estimation**: find the most likely parameter settings under the posterior

- Two approaches to classification: **Discriminative** and **Generative**

  Discriminative: directly learn to predict t as a function of x. Generative: model the data distribution for each class separately, and make predictions using posterior inference.

- **Bayes classifier**: given features x, we compute the posterior class probabilities using Bayes’ Rule

- **Naı̈ve Bayes** makes the assumption that the word features x j are **conditionally independent** given the class t. This means x_i and x_j are independent under the conditional distribution p(x|t). This doesn’t mean they’re independent.

- **Gaussian Discriminant Analysis** in its general form assumes that p(x|t) is distributed according to a multivariate Gaussian distribution

  We can go even further and assume the covariances are spherical, or **isotropic**

- The more interesting case is when some of the variables are latent, or never observed. These are called **latent variable models**. 

  If t is never observed, we call it a **latent variable**, or **hidden variable**, and generally denote it with z instead.

  The things we can observe (i.e. x) are called **observables**.

  p(x) = Σ p(x,z) = Σ p(x|z) p(z)

  f p(z) is a categorial distribution, this is a **mixture model**, and different values of z correspond to different **components**.

- Warning: you don’t want the global maximum. You can achieve arbitrarily high training likelihood by placing a small-variance Gaussian component on a training example.
  This is known as a **singularity**.



**Comparison** of maximum likelihood and Bayesian parameter estimation

- Some advantages of the Bayesian approach 

  - More robust to data sparsity

  - Incorporate prior knowledge

  - Smooth the predictions by averaging over plausible explanations
- Problem: maximum likelihood is an optimization problem, while Bayesian parameter estimation is an integration problem
    - This means maximum likelihood is much easier in practice, since we can just do gradient descent
    - Automatic differentiation packages make it really easy to compute gradients
    - There aren’t any comparable black-box tools for Bayesian parameter estimation (although Stan can do quite a lot)



**Naı̈ve Bayes** is an amazingly cheap learning algorithm!

- Training time: estimate parameters using maximum likelihood
  - Compute co-occurrence counts of each feature with the labels.
  - Requires only one pass through the data!

- Test time: apply Bayes’ Rule
  - Cheap because of the model structure. (For more general models, Bayesian inference can be very expensive and/or complicated.)

- Unfortunately, it’s usually less accurate in practice compared to discriminative models.
  - The problem is the “naı̈ve” independence assumption.
  - We’re covering it primarily as a stepping stone towards latent variable
    models.



When should we prefer **GDA** to **LR**, and vice versa?

- GDA makes a stronger modeling assumption: assumes class-conditional data is multivariate Gaussian 
  - If this is true, GDA is asymptotically efficient (best model in limit of large N)
  - If it’s not true, the quality of the predictions might suffer.

- Many class-conditional distributions lead to logistic classifier.
  - When these distributions are non-Gaussian (i.e., almost always), LR usually beats GDA

- GDA can handle easily missing features (how do you do that with LR?)



## 隐私

**Randomized Response**

指对于特定输入，该算法的输出不是固定值，而是服从某一分布

**Neighbouring Database**

定两个数据集D和D’, 若它们有且仅有一条数据不一样，那我们就称此二者为相邻数据集。以上面数据集为例：假定有 ![[公式]](https://www.zhihu.com/equation?tex=n) 个人，他们是否是单身狗，形成一个集合 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Ba_1%2Ca_2%2C+%E2%80%A6%2C+a_n%5C%7D) （其中 ![[公式]](https://www.zhihu.com/equation?tex=a_i+%3D+0)或![[公式]](https://www.zhihu.com/equation?tex=1)），那么另一个集合当中只有一个人改变了单身状态，形成另一个集合 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Ba_1%E2%80%99%2C+a_2%E2%80%99%2C+%E2%80%A6%2C+a_n%E2%80%99%5C%7D) ，也就是只存在一个 ![[公式]](https://www.zhihu.com/equation?tex=i) 使得 ![[公式]](https://www.zhihu.com/equation?tex=a_i+%5Cne+a_i%E2%80%99) ，那么这两个集合便是相邻集合。

The way in which the curator responds to queries is called the **mechanism**.

**Differential Privacy**

差分隐私形式化的定义为：

![[公式]](https://www.zhihu.com/equation?tex=Pr%5C%7BA%28D%29+%3D+O%5C%7D+%E2%89%A4e%5E%5Cepsilon+%5Ccdot+Pr%5C%7BA%28D%E2%80%99%29+%3D+O%5C%7D+) 

也就是说，如果该算法作用于任何相邻数据集，得到一个特定输出 ![[公式]](https://www.zhihu.com/equation?tex=O) 的概率应差不多，那么我们就说这个算法能达到差分隐私的效果。也就是说，观察者通过观察输出结果很难察觉出数据集一点微小的变化，从而达到保护隐私的目的。

### Laplace Mechanism

**Gaussian noise** violates our definition, but only because of the tails. It satisfies a different definition of differential privacy which allows violating the ε constraint with small probability, but that’s beyond the scope of this lecture. (gap increase linearly)

**Laplace mechanism**: return a vector y whose entries are independently sampled from Laplace distributions. 

Claim: the **Laplace mechanism** is differentially private.

### Exponential  mechanism

指数机制（The exponential  mechanism）是为我们希望选择“最佳”响应的情况而设计的，但直接在计算数量上添加噪声可以完全破坏其价值。例如在拍卖中设定价格，其目标是最大化收益，以及在最优价格上添加少量正噪声（为了保护投标的隐私）可以大大减少由此产生的收入。



### MAP and Full Bayesian 

Full Bayesian inference means that you learn a full posterior distribution ...
MAP is simply mode of the posterior distribution. They can’t be the  same since mode of the distribution is not the same as distribution  itself. MAP is a point estimate, while full posterior is a distribution  estimate.

As seen above, Bayesian inference provides much more information than  point estimators like MLE and MAP. However, it also has a drawback — the complexity of its integral computation. The case in this article was  quite simple and solved analytically, but it’s not always the case in  real-world applications. We then need to use MCMC or other algorithms as a substitute for the direct integral computation. 
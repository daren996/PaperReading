# DecisionTree

--------------------------

## Approaches

Three common division methods of classification decision tree (algorithms used to develop decision trees) are:

1. ID3, Information Gain. 

   <img src="http://latex.codecogs.com/gif.latex?g(D,A)=H(D)-H(D|A)" />

2. C4.5, Ratio of Information Gain. 

   <img src="http://latex.codecogs.com/gif.latex?g_R(D,A)=\frac{g(D,A)}{H_A(D)}" />

3. CART, Gini Coefficient. 

   <img src="http://latex.codecogs.com/gif.latex?Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2)" />  



## Realize Decision Tree in Sklearn.

[Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

[Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html): Convert a collection of text documents to a matrix of token counts. 

Model Selection. 

1. [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split): split arrays or matrices into random train and test subsets. 
2. [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score): evaluate a score by cross-validation. 



## History 

1. 1948 克劳德·香农介绍了信息论，这是决策树学习的基础之一。

   Shannon, C.E. (1948).A mathematical theory of communication*.* Bell System Technical Journal, 27: 379–423 and 623–656. //Shannon, C. E. (1949). Communication Theory of Secrecy Systems. Bell System Technical Journal 28 (4): 656–715.

2. 1963 Morgan和Sonquist开发出第一个回归树。 

   Morgan. J. N. & Sonquist, J. A. (1963) Problems in the Analysis of Survey Data, and a Proposal, Journal of the American Statistical Association, 58:302, 415-434.

3. 1980 Gordon V. Kass开发出CHAID算法。 

   Gordon V. K. (1980).An Exploratory Technique for Investigating Large Quantities of Categorical Data, Applied Statistics. 29(2): 119–127.

4. 1984 Leo Breiman et al.开发出CART（分类与回归树）。 

   Breiman, L.; Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). Classification and regression trees. Monterey, CA: Wadsworth & Brooks/Cole Advanced Books & Software.

5. 1986 Quinlan开发出ID3算法。 

   Quinlan, J. R. (1986). Induction of Decision Trees. Mach. Learn. 1(1): 81–106

6. 1993 Quinlan开发出C4.5算法。 

   Quinlan, J. R. (1993). C4.5: Programs for machine learning. San Francisco,CA: Morgan Kaufman.

7. 1997 Loh和Shih开发出QUEST。 

   Loh, W. Y., & Shih, Y. S. (1997). Split selection methods for classification trees. Statistica sinica, 815-840.

8. 1999 Yoav Freund和Llew Mason提出AD-Tree。 

   Freund, Y., & Mason, L. (1999, June). The alternating decision tree learning algorithm. In icml (Vol. 99, pp. 124-133).

9. 2017 Geoffrey Hinton等人发表arXiv论文提出「软决策树」（Soft Decision Tree）。 

   Frosst, N.; Hinton, G. (2017).Distilling a Neural Network Into a Soft Decision Tree.arXiv:1711.09784.

10. 2018 加州大学洛杉矶分校的朱松纯教授等人发布了一篇使用决策树对CNN的表征和预测进行解释的论文。

    Zhang, Q.; Yang, Y.; Nian Wu, Y.; Zhu, S-C. (2018). Interpreting CNNs via Decision Trees. arXiv:1802.00121.



## Random Forests

Random forests = bagged decision trees, with one extra trick to decorrelate the predictions.

> When choosing each node of the decision tree, choose a random set of d input features, and only consider splits on those features
> Random forests are probably the best black-box machine learning algorithm — they often work well with no tuning whatsoever. One of the most widely used algorithms in Kaggle competitions.

## Advantage and Drawbacks

### Advantages of decision trees over KNN

1. Good when there are lots of attributes, but only a few are important Good with discrete attributes
2. Easily deals with missing values (just treat as another value).
3. Robust to scale of inputs.
4. Fast at test time.
5. More interpretable.

### Advantages of KNN over decision trees

1. Few hyperparameters.
2. Able to handle attributes/features that interact in complex ways (e.g. pixels). 
3. Can incorporate interesting distance measures (e.g. shape contexts). 
4. Typically make better predictions in practice. As we’ll see next lecture, ensembles of decision trees are much stronger. But they lose many of the advantages listed above.

# DecisionTree

--------------------------

## Realize Decision Tree in Sklearn.

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

# DecisionTree

--------------------------

## Realize Decision Tree in Sklearn.

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

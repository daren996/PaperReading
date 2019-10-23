# Entity Resolution Explanations



## 1. Model Explainer

### 1.1 Why do we need Model Explanations?

Machine learning models remain mostly black boxes. Understanding the reasons behind predictions is quite important in assessing **trust**, which is fundamental if one plans to take action based on a prediction, or when choosing whether to deploy a new model. If the users do not trust a model or a prediction, they will not use it. Understanding will transform an untrustworthy model or prediction into a trustworthy one by looking insights into the model. 

It is important to differentiate between two different (but related) definitions of **trust**: (1) trusting a prediction, i.e. whether a user trusts an individual prediction sufficiently to take some action based on it; (2) trusting a model, i.e. whether the user trusts a model to behave in reasonable ways if deployed. Both are directly impacted by how much the human understands a model’s behaviour. 

Determining trust in individual **predictions** is an important problem when the model is used for decision making. In some cases, predictions cannot be acted upon on blind faith, as the consequences may be catastrophic, such as using machine learning for medical diagnosis or terrorism detection.

Apart from trusting individual predictions, there is also a need to evaluate the **model** as a whole before deploying it “in the wild”. To make this decision, users need to be confident that the model will perform well on real-world data. 

### 1.2 Direction

**Model Trust: Metrics -> Explained Examples**

For the trust a whole **model**, currently, models are evaluated using accuracy **metrics** on an available validation dataset. A traditional pipelnine often developes a classification model using annotated data, of which a held-out subset is used for automated evaluation. However, real-world data is often significantly different, as practitioners often overestimate the accuracy of their models, and further, the evaluation metric may not be indicative of the product’s goal. Thus, trust cannot rely solely on it. 

Inspecting individual predictions (**examples**) and their explanations is a worthwhile solution. In this case, it is important to aid users by suggesting which instances to inspect, especially for large datasets. 

**Drawbacks of Metrics**

Machine learning practitioners often have to select a model from a number of alternatives, requiring them to assess the relative trust between two or more models.  In addition to the drawbacks of metrics mentioned above, it may also cause the machine learning practitioners to have a wrong assessment of the model, and the explanations can correct this assessment.  

Here is the comparison of a variety of models with the help of explanations. 

![compare-accuracy-explanations](https://github.com/daren996/PaperReading/blob/master/MOD/Images/compare-accuracy-explanations.png)

The right side is a support vector machine with RBF kernel trained on unigrams to differentiate “Christianity” from “Atheism” (on a subset of the 20 newsgroup dataset). Although this classifier achieves 94% held-out accuracy, and one would be tempted to trust it based on this, the explanation for an instance shows that predictions are made for quite arbitrary reasons (words “Posting”, “Host”, and “Re” have no connection to either Christianity or Atheism). The word “Posting” appears in 22% of examples in the training set, 99% of them in the class “Atheism”. 

In this case, the algorithm with higher accuracy on the validation set is actually much **worse**, a fact that is easy to see when explanations are provided (again, due to human prior knowledge), but hard otherwise. Further, there is frequently a **mismatch** between the metrics that we can compute and optimize (e.g. accuracy) and the actual metrics of interest such as user engagement and retention. While we may not be able to measure such metrics, we have knowledge about how certain model behaviors can influence them. 

Therefore, a practitioner may wish to choose a less accurate model for content recommendation that does not place high importance in features related to “clickbait” articles (which may hurt user retention), even if exploiting such features increases the accuracy of the model in cross validation. We note that explanations are particularly useful in these (and other) scenarios if a method can produce them for any model, so that a variety of models can be compared. [Clickbait-Article](https://github.com/daren996/PaperReading/blob/master/MOD/Images/clickbait-news.png)

### 1.3 Characteristics of wrong models and evaluations

**Data Leakage** The unintentional leakage of signal into the training (and validation) data that would not appear when deployed. Still, consider the example of automated diagnosis, a model finds that the patient ID is heavily correlated with the target class in the training and validation data. This issue would be incredibly challenging to identify just by observing the predictions and the raw data, so we want to solve this problem by explaining the model and predictions.

**Dataset Shift** The training data is different than the test data. For example in the famous 20 newsgroups dataset, 

The insights given by expla- nations are particularly helpful in identifying what must be done to convert an untrustworthy model into a trustworthy one - for example, removing leaked data or changing the training data to avoid dataset shift. 

## 2. "Why Should I Trust You?" Explaining the Predictions of Any Classifier (2016)

### 2.1 Structure

**LIME:** Explanation of predicts of any classifier or regressor. 
**SP-LIME:** Explanation of models by selecting representation instances. 

### 2.2 Example: Diagnosis

In this case, an explanation is a small list of symptoms with relative weights - symptoms that either contribute to the prediction (in green) or are evidence against it (in red). Humans usually have prior knowledge about the application domain, which they can use to accept (trust) or reject a prediction if they understand the reasoning behind it. It is clear that a doctor is much better positioned to make a decision with the help ol a model if intelligible explanations are provided. 

It has been observed, for example, that providing explanations can increase the acceptance of movie recommendations and other automated systems. 

![process-explanation-diagnosis](https://github.com/daren996/PaperReading/blob/master/MOD/Images/process-explanation-diagnosis.png)

In this picture, a model predicts that a patient has the flu, and **LIME** highlights the symptoms in the patient’s history that led to the prediction. Sneeze and headache are portrayed as contributing to the “flu” prediction, while “no fatigue” is evidence against it. With these, a doctor can make an informed decision about whether to trust the model’s prediction. 

As we have talked about previously, **data leakage**, where a model finds that the patient ID is heavily correlated with the target class, can be solved easier if explanations such as the one in the above figure are provided, as patient ID would be listed as an explanation for predictions. 

### 2.3 LIME

The overall goal of LIME is to identify an **interpretable model** over the **interpretable representation** that is **locally faithful** to the classifier. 

**Interpretable Data Representations** 

Features: Actually used by the model. Interpretable representations: Understandable to humans. 

Original representation: <img src="http://latex.codecogs.com/gif.latex?x \in R^d" />. Interetable representation: <img src="http://latex.codecogs.com/gif.latex?x' \in \{0, 1\}^{d'}" />. 

For text classification, a possible interpretable representation is a binary vector indicating the presence or absence of a word, even though the classifier may use more complex (and incomprehensible) features such as word embeddings.

For image classification, an interpretable representation may be a binary vector indicating the “presence” or “absence” of a contiguous patch of similar pixels (a superpixel), while the classifier may represent the image as a tensor with three color channels per pixel.  

(need two pictures here)

**Formulation** 

Let the model being explained be denoted <img src="http://latex.codecogs.com/gif.latex?\ f: R^d \rightarrow R" />.  (In classification, f(x) is the probability that x belongs to a certain class.) We further use <img src="http://latex.codecogs.com/gif.latex?\ \pi_x(z)" /> as a proximity measure between an instance z to x, so as to define locality around x. 

We define an explanation as a model g ∈ G, where **explanation families** G is a class of potentially interpretable models, such as linear models, decision trees, or falling rule lists. (They can be presented to the user with visual or textual artifacts.) The domain of g is <img src="http://latex.codecogs.com/gif.latex?\{0, 1\}^{d'}" />, that is the absence/presence of the interpretable components.

**Complexity measures** <img src="http://latex.codecogs.com/gif.latex?\Omega(g)" /> represents the measure of complexity of explanation g ∈ G. (The higher it is, the lower interpretability it has. For a decision tree, the Ω(g) could be the depth of the tree; for linear models, Ω(g) may be the number of non-zero weights.)

Let **fidelity function** <img src="http://latex.codecogs.com/gif.latex?\ L(f, g, \pi_x)" /> be a measure of how unfaithful g is in approximating f in the locality defined by <img src="http://latex.codecogs.com/gif.latex?\ \pi_x" />. At the same time, Ω(g) should also be low enough to be interpretable by humans. The finally explanation produced by LIME is obtained by: 

<img src="http://latex.codecogs.com/gif.latex?\ \xi(x) = \mathop{\arg\min}_{g \in G} L(f, g, \pi_x) + \Omega(g)" />



## Reference

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016, August). **Why should i trust you?: Explaining the predictions of any classifier.** In *Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining* (pp. 1135-1144). ACM.


	@inproceedings{ribeiro2016should,
	  title={Why should i trust you?: Explaining the predictions of any classifier},
	  author={Ribeiro, Marco Tulio and Singh, Sameer and Guestrin, Carlos},
	  booktitle={Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining},
	  pages={1135--1144},
	  year={2016},
	  organization={ACM}
	}


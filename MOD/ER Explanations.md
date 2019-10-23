# Entity Resolution Explanations



## Model Explainer

### Why do we need Model Explanations?

Machine learning models remain mostly black boxes. Understanding the reasons behind predictions is quite important in assessing **trust**, which is fundamental if one plans to take action based on a prediction, or when choosing whether to deploy a new model. If the users do not trust a model or a prediction, they will not use it. Understanding will transform an untrustworthy model or prediction into a trustworthy one by looking insights into the model. 

It is important to differentiate between two different (but related) definitions of **trust**: (1) trusting a prediction, i.e. whether a user trusts an individual prediction sufficiently to take some action based on it; (2) trusting a model, i.e. whether the user trusts a model to behave in reasonable ways if deployed. Both are directly impacted by how much the human understands a model’s behaviour. 

Determining trust in individual **predictions** is an important problem when the model is used for decision making. In some cases, predictions cannot be acted upon on blind faith, as the consequences may be catastrophic, such as using machine learning for medical diagnosis or terrorism detection.

Apart from trusting individual predictions, there is also a need to evaluate the **model** as a whole before deploying it “in the wild”. To make this decision, users need to be confident that the model will perform well on real-world data. 

### Direction

**Model Trust: Metrics -> Explained Examples**

For the trust a whole **model**, currently, models are evaluated using accuracy **metrics** on an available validation dataset. A traditional pipelnine often developes a classification model using annotated data, of which a held-out subset is used for automated evaluation. However, real-world data is often significantly different, as practitioners often overestimate the accuracy of their models, and further, the evaluation metric may not be indicative of the product’s goal. Thus, trust cannot rely solely on it. 

Inspecting individual predictions (**examples**) and their explanations is a worthwhile solution. In this case, it is important to aid users by suggesting which instances to inspect, especially for large datasets. 

**Drawbacks of Metrics**

Machine learning practitioners often have to select a model from a number of alternatives, requiring them to assess the relative trust between two or more models.  In addition to the drawbacks of metrics mentioned above, it may also cause the machine learning practitioners to have a wrong assessment of the model, and the explanations can correct this assessment.  

Here is the comparison of a variety of models with the help of explanations. 

![compare-accuracy-explanations](https://github.com/daren996/PaperReading/blob/master/MOD/Images/compare-accuracy-explanations.png)

In this case, the algorithm with higher accuracy on the validation set is actually much **worse**, a fact that is easy to see when explanations are provided (again, due to human prior knowledge), but hard otherwise. Further, there is frequently a **mismatch** between the metrics that we can compute and optimize (e.g. accuracy) and the actual metrics of interest such as user engagement and retention. While we may not be able to measure such metrics, we have knowledge about how certain model behaviors can influence them. 

Therefore, a practitioner may wish to choose a less accurate model for content recommendation that does not place high importance in features related to “clickbait” articles (which may hurt user retention), even if exploiting such features increases the accuracy of the model in cross validation. We note that explanations are particularly useful in these (and other) scenarios if a method can produce them for any model, so that a variety of models can be compared. [Clickbait-Article](https://github.com/daren996/PaperReading/blob/master/MOD/Images/clickbait-news.png)

### Characteristics of wrong models and evaluations

**Data Leakage** The unintentional leakage of signal into the training (and validation) data that would not appear when deployed. Still, consider the example of automated diagnosis, a model finds that the patient ID is heavily correlated with the target class in the training and validation data. This issue would be incredibly challenging to identify just by observing the predictions and the raw data, so we want to solve this problem by explaining the model and predictions.

**Dataset Shift** The training data is different than the test data. For example in the famous 20 newsgroups dataset, 

The insights given by expla- nations are particularly helpful in identifying what must be done to convert an untrustworthy model into a trustworthy one - for example, removing leaked data or changing the training data to avoid dataset shift. 

## "Why Should I Trust You?" Explaining the Predictions of Any Classifier (2016)

### Structure

**LIME:** Explanation of predicts of any classifier or regressor. 
**SP-LIME:** Explanation of models by selecting representation instances. 

### Example: Diagnosis

In this case, an explanation is a small list of symptoms with relative weights - symptoms that either contribute to the prediction (in green) or are evidence against it (in red). Humans usually have prior knowledge about the application domain, which they can use to accept (trust) or reject a prediction if they understand the reasoning behind it. It is clear that a doctor is much better positioned to make a decision with the help ol a model if intelligible explanations are provided. 

It has been observed, for example, that providing explanations can increase the acceptance of movie recommendations and other automated systems. 

![process-explanation-diagnosis](https://github.com/daren996/PaperReading/blob/master/MOD/Images/process-explanation-diagnosis.png)

In this picture, a model predicts that a patient has the flu, and **LIME** highlights the symptoms in the patient’s history that led to the prediction. Sneeze and headache are portrayed as contributing to the “flu” prediction, while “no fatigue” is evidence against it. With these, a doctor can make an informed decision about whether to trust the model’s prediction. 

As we have talked about previously, **data leakage**, where a model finds that the patient ID is heavily correlated with the target class, can be solved easier if explanations such as the one in the above figure are provided, as patient ID would be listed as an explanation for predictions. 

### LIME

The overall goal of LIME is to identify an **interpretable model** over the **interpretable representation** that is **locally faithful** to the classifier. 

**Interpretable Data Representations** 
Features: Actually used by the model. 
Interpretable representations: Understandable to humans.



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


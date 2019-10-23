# Entity Resolution Explanations



## Background

### Why do we need ER Explanations?

Machine learning models remain mostly black boxes. Understanding the reasons behind predictions is quite important in assessing **trust**, which is fundamental if one plans to take action based on a prediction, or when choosing whether to deploy a new model. If the users do not trust a model or a prediction, they will not use it. Understanding will transform an untrustworthy model or prediction into a trustworthy one by looking insights into the model. 

It is important to differentiate between two different (but related) definitions of **trust**: (1) trusting a prediction, i.e. whether a user trusts an individual prediction sufficiently to take some action based on it; (2) trusting a model, i.e. whether the user trusts a model to behave in reasonable ways if deployed. Both are directly impacted by how much the human understands a model’s behaviour. 

Determining trust in individual **predictions** is an important problem when the model is used for decision making. In some cases, predictions cannot be acted upon on blind faith, as the consequences may be catastrophic, such as using machine learning for medical diagnosis or terrorism detection.

Apart from trusting individual predictions, there is also a need to evaluate the **model** as a whole before deploying it “in the wild”. To make this decision, users need to be confident that the model will perform well on real-world data. 

### Direction

For the trust a whole **model**, currently, models are evaluated using accuracy **metrics** on an available validation dataset. However, real-world data is often significantly different, and further, the evaluation metric may not be indicative of the product’s goal. Inspecting individual predictions and their explanations is a worthwhile solution, in addition to such metrics. In this case, it is important to aid users by suggesting which instances to inspect, especially for large datasets.



## LIME

**Why should i trust you?: Explaining the predictions of any classifier. (2016)** 

### Structure

**LIME:** Explanation of predicts of any classifier or regressor. 
**SP-LIME:** Explanation of models by selecting representation instances. 

### Example: Diagnosis

In this case, an explanation is a small list of symptoms with relative weights - symptoms that either contribute to the prediction (in green) or are evidence against it (in red). Humans usually have prior knowledge about the application domain, which they can use to accept (trust) or reject a prediction if they understand the reasoning behind it. It is clear that a doctor is much better positioned to make adedceiscioisnion with the help ol a model if intelligible explanations are provided. 

It has been observed, for example, that providing explanations can increase the acceptance of movie recommendations and other automated systems. 

![image-20191022214116310](https://github.com/daren996/PaperReading/blob/master/MOD/Images/process-explanation-diagnosis.png)



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


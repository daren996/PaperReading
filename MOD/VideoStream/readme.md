# video streams



1. [Papers](#papers)
2. [Reports](#reports)



# Papers

### Surveys

#### A Comprehensive Survey of Vision-Based Human Action Recognition Methods

Data: GRB and Depth & Sketleton

Human Action Recognition:
- Classification
  - Action Representation
    - Handcraft
    - Deep Learning: Two-Stream CNN; LSTM; 3D CNN
  - Interaction Recognition
- Detection
  - Action Detection



### Detecting and Recognizing Human-Object Interaction

- Get <human, verb, object>
- InteractNet

### Other directions

learn action or poses from video clips.



# Reports

This is also a research class project of [CSC 2508](https://koudas.github.io/csc2508-fall2019.html): Working on a research project and file a single submission. The project will be broken down to three assignments: (1) initial research proposal, (2) Intermediate report, (3) final report and presentation.

##  Project Proposal

Oct 21st, 2019

- Topic to be addressed and the nature of the problem.
- State of the art (prior work, what remains unsolved, etc.)
- The proposed technique to be implemented/evaluated.
- To what degree the project will repeat existing work.
- Specific, measurable goals: deliverables, and dates you expect to produce them.

### Problem Statement

Recent advances in computer vision - in the form of deep neural networks - have made it possible to query increasing volumes of video data with high accuracy. However, neural network inference is computationally expensive at scale. Although some have proposed systems for accelerating neural network queries over the video, there are still many limitations for them to be applied to real-world scenarios. 

To understand the visual world, a machine must not only recognize individual object instances but also how they interact. Most of the state-of-the-art video query processing models focus on the queries about the count of categorized objects and their location relationships. Semantic information, such as human-object interaction information, is often ignored. 

I address the task of queries for video stream database system, about not only the detection of objects and categories and their locations but also semantic information such as human-object interaction. For queries involving actions/poses, the algorithm can, firstly, quickly detect the humans and objects that appear in each frame, as well as their location information, and then return the most likely combination of triplet, <human, action, object>, in these frames. 

### Prior work

#### Data Management on Video Streams

Recently there has been increased interest in the application of Deep Learning techniques in data management. Many works tried to apply image classification and object detection algorithms to fast query processing on video streams.

NoScope [2] uses a modified form of distillation [4] to train an extremely lightweight specialized model at the cost of generalization to other videos. Given a target video, object to detect, and reference neural network, NoScope automatically searches for and trains a sequence of models that preserves the accuracy of the reference network but is specialized to the target video and is therefore far less computationally expensive. 

Based on NoScope, [3] has proposed a system, BlazeIt, that optimizes queries over video for spatiotemporal information of objects. BlazeIt accepts queries via a declarative language, FrameQL, and new query optimization techniques including an aggregation algorithm, a scrubbing algorithm, and a selection algorithm to leverage imprecise specialized NNs, find rare events and apply content-based selection. 

[1] presented a series of filters to estimate the number of objects in a frame, the number of objects of specific classes in a frame as well as to assess an estimate of the spatial position of an object in a frame. Although these algorithms have achieved good accuracy for counting and location estimation purposes and dramatic speedups by several orders of magnitude in real video datasets, there are still many additional query types that need to be considered. 

#### Detecting and Recognizing Human-object Interaction

Visual recognition of individual instances, e.g., detecting objects and estimating human actions has witnessed significant improvements thanks to deep learning visual representations. [5] proposed a novel model that is driven by a human-centric approach, which learns to predict an action-specific density over target object locations based on the appearance of a detected person using the appearance of a person as a cue for localizing the objects they are interacting with.

Some works used video clips for action/pose recognition. I will review them in the future, as I might use them in my approach. however, these NN-based algorithms cannot be applied to video stream queries directly due to their expensive at scale. 

### Approach/Methodology

#### Proposed Techniques

For queries involving actions/poses of humans, the algorithm can, first, quickly detect the humans and objects that appear in each frame, as well as their location information, and then return the most likely combination of triplet, <human, action, object>, in these frames. 

There are many ways to implement the above algorithm. First, a neural network similar to human-centric branch [5] can be applied to obtain the score of assigning an action for the person and whether the object is the actual target of the action; then, the whole model obtains the actions that may exist in each frame, based on the above scores. Second, actions are made up of many consecutive frames; thus, detecting human-object interaction in the video stream can be based on extracted video clips. Based on this idea, we can propose a video clips extraction algorithm and an action detection algorithm on video clips, so that the return of queries can be multiple video clips, each of which belongs to a <human, action, object> triplet.

#### To what degree the project will repeat existing work

I will refer to the definition of SQL-like language of video stream queries in prior works. When implementing object detection and the action detection part, we will learn from the models in [1, 2, 3, 6] and [5] respectively. However, the network of object detection in previous work is complicated and can not be directly applied to the current scenario. Therefore, the modification of the model so that it can better meet the needs of video stream queries will be the main contribution of this project.

### Plan of Action
- Literature Review about detecting and recognizing semantic information in images. Oct 21 - Oct 27.

  The main task at this stage is to find models that are suitable for use in video queries involving semantic information and think about how to improve them.

- Discussed the models with Professor Koudas. Oct 28.

- Write and run the code of previous works. Oct 28 - Nov 3.

- Write the code of improved model. Nov 3 - Nov 10.

  Also try to find related data sets.

- Discussed with professor Koudas. Nov 11.

- Experiments. Nov 12 - Nov 27.

  The model should be evaluated and adjusted during this time.

### Reference

**[1]** Koudas, N., Li, R., Xarchakos, I.. (2020). Video Monitoring Queries. Under Review.

**[2]** Kang, D., Emmons, J., Abuzaid, F., Bailis, P., & Zaharia, M. (2017). Noscope: optimizing neural network queries over video at scale. *Proceedings of the VLDB Endowment*, *10*(11), 1586-1597.

**[3]** Kang, D., Bailis, P., & Zaharia, M. (2018). Blazeit: Fast exploratory video queries using neural networks. *arXiv preprint arXiv:1805.01046*.

**[4]** Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*.

**[5]** Gkioxari, G., Girshick, R., Dollár, P., & He, K. (2018). Detecting and recognizing human-object interactions. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 8359-8367).

**[6]** Redmon, J., & Farhadi, A. (2017). YOLO9000: better, faster, stronger. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 7263-7271).



```
@article{koudas2020video,
  title={Video Monitoring Queries},
  author={Nick Koudas, Raymond Li, Ioannis Xarchakos},
  journal={Under Review},
  year={2020}
}
@article{kang2017noscope,
  title={Noscope: optimizing neural network queries over video at scale},
  author={Kang, Daniel and Emmons, John and Abuzaid, Firas and Bailis, Peter and Zaharia, Matei},
  journal={Proceedings of the VLDB Endowment},
  volume={10},
  number={11},
  pages={1586--1597},
  year={2017},
  publisher={VLDB Endowment}
}
@article{kang2018blazeit,
  title={Blazeit: Fast exploratory video queries using neural networks},
  author={Kang, Daniel and Bailis, Peter and Zaharia, Matei},
  journal={arXiv preprint arXiv:1805.01046},
  year={2018}
}
@article{hinton2015distilling,
  title={Distilling the knowledge in a neural network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal={arXiv preprint arXiv:1503.02531},
  year={2015}
}
@inproceedings{gkioxari2018detecting,
  title={Detecting and recognizing human-object interactions},
  author={Gkioxari, Georgia and Girshick, Ross and Doll{\'a}r, Piotr and He, Kaiming},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={8359--8367},
  year={2018}
}
@InProceedings{redmon2017yolo9000,
  author = {Redmon, Joseph and Farhadi, Ali},
  title = {YOLO9000: Better, Faster, Stronger},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {July},
  year = {2017}
}
```

##  Progress Report

Nov 11th, 2019

### Progress Description

Video Stream Interaction Queries is under development. It currently incorporates the layout of [3] with few new features. Our layout, will allow users to express human object interaction queries and compare the time and accuracy performance of our approach with the full model's performance [1]. Figure 1 presents the current state of the front end. 

<img src='https://github.com/daren996/PaperReading/blob/master/MOD/VideoStream/img/demo.png' width=80%>

In comparison with [3], we moved the query and video source selection on the side bar. Subsequently, we use the extra space to present the filters' ordering per batch size on the upper right corner. The filters' ordering will be updated every batch size. We plan to add an extra selection in order the user to define the batch size. Each color represents a different operator, and a node represents a filter of an operator.
Figure 2 presents the filters ordering module.

<img src='https://github.com/daren996/PaperReading/blob/master/MOD/VideoStream/img/filters.png' width=40%>

The SQL Query area on upper left corner, it will output an SQL action query based on the query definition. By clicking on Optimisation button, our approach will be invoked, described on **Querying For Interactions**, while clicking Brute Force, state of the art object detectors [2] and action recognition [1] models will evaluate the respective input video. 

Finally, the video display for both our approach and full model's approach will produce bounding boxes for humans and objects which participate on an action. Additionally, we will produce a bounding box or a heat-map for frames that a specific action exist between the human and the object.

### Plan for Future Work

- Nov 11 Discussing the new model and additional features that can be added.

- Nov 11 - Nov 15 Implement modules to present the model’s accuracy and time performance.

- Nov 11 - Nov 25 Writing the code of the improved model using human action data sets.

- Nov 25 - Dec 1 Preparing the report and presentation.

### Reference

**[1]**  Georgia Gkioxari, Ross Girshick, Piotr Doll ́ar, and Kaiming He. Detecting and recognizing human-object interactions.  In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8359–8367, 2018.

**[2]**  K. He, G. Gkioxari, P. Doll ́ar, and R. Girshick. Mask r-cnn. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 2980–2988, Oct 2017.

**[3]**  Ioannis  Xarchakos  and  Nick  Koudas. Svq: Streaming  video  queries. In Proceedings of the 2019 International Conference on Management of Data, SIGMOD ’19, pages 2013–2016, New York, NY, USA, 2019. ACM.
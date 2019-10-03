# Query Optimization



## What is Query Optimization?





## Papers

- Towards A Hands Free Query Optimizer with Deep Learning
- NEO: A Learned Query Optimizer
- Learning State Representations for Query Optimization With Deep Reenforcement Learning 
- Deep Reinforcement Learning for join order enumeration aiDM 2018 



## Questions:

- Learning State Representations for Query Optimization with Deep Reinforcement Learning

1. What is the meaning of Cardinality, especially in the context of the representation of state and action? If it represents the cost (or latency) of a specific state and action, could you please give me an example?

2. In section 3.1, how can hx represent a query and how can it change through NNST? Is there any example of hx? 

3. In section 3.1, the authors said that, for traditional methods, the size of the feature vector would have to grow with the complexity of databases and queries. How do the new methods overcome this difficulty? 

- Deep Reinforcement Learning for Join Order Enumeration

1. In the ReJOIN model, the reward is the reciprocal of the cost. How does the model calculate the reward (or cost)?

2. Figure 5 shows that the running time of the model does not increase linearly with the increase in relation. However, since only one action is selected for each iteration of the model, the number of iterations will increase with the increase in relation. Why does the actual running time not increase linearly?

3. In Figure 3, Why is the value in the rightmost Tree Vector 1/4? The height of the tree is 3, should not the value of tree vector be 1/3?

- Towards a Hands-Free Query Optimizer through Deep Learning

1. In section 3, the bottom-up nature of ReJOIN’s algorithm is O(n). Why does Figure 3 show that the time overhead of ReJOIN barely grows or even fluctuates as the relation increases?

2. In section 5.2, the cost model bootstrapping methods struggle to cope with the switch of units. When does the switch of units generally take place?

3. In section 5.3.1, how does the knowledge gained from the previous training phase help the model train significantly faster in subsequent phases? 

- Neo: A Learned Query Optimizer

1. In the DNN-guided plan search, will the heap become such big as the number of relations increases? There is much information on other subquery stored in the heap. What is the point of storing this information?
2. Is the DNN-guided plan search still shortsighted? If yes, how can it be proved to find the best or better results?
3. For the column predicate vector of the query encoding, how does the histogram work? 



## Reinforcement Learning


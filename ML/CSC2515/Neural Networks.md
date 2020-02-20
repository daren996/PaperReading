## Neural Networks

All input units are connected to all output units. We call this a **fully connected layer**.

A multilayer network consisting of fully connected layers is called a **multilayer perceptron**. 

Multilayer feed-forward neural nets with nonlinear activation functions are **universal function approximators**: they can approximate any function arbitrarily well.

Limits of **universality**: 

1. You may need to represent an exponentially large network. 
2. If you can learn any function, you’ll just overfit.
3. Really, we desire a compact representation! 

#### 1. Backpropagation 

**Computation graph**: The nodes represent all the inputs and computed quantities, and the edges represent which nodes are computed directly as a function of which other nodes. 

**Multivariate Chain Rule** 

**Computational Cost**: the backward pass is about as expensive as two forward passes. 

#### 2. Gradient Checking

Check your derivatives numerically by plugging in a small value of h, This is known as **finite differences**.

Gradient checking is really important!

Learning algorithms often appear to work even if the math is wrong. But: 

1. They might work much better if the derivatives are correct. 
2. Wrong derivatives might lead you on a wild goose chase. 

#### 3. Convex

Unfortunately, training a network with hidden units cannot be convex because of permutation symmetries. We can re-order the hidden units in a way that preserves the function computed by the network. 

Because of permutation symmetry, there are K! permutations of the hidden units in a given layer which all compute the same function. 

Suppose we average the parameters for all K! permutations. Then we get a degenerate network where all the hidden units are identical. 

If the cost function were convex, this solution would have to be better than the original one, which is ridiculous! 

Hence, training multilayer neural nets is non-convex. 

#### 4. Convolution 

The thing we convolve by is called a **kernel**, or **filter**. 

What do these filters do? 

	# Blur
	0 1 0
	1 4 1
	0 1 0
	
	# Sharppen
	0 -1  0
	-1 8 -1
	0 -1  0
	
	# Edge
	0 -1  0
	-1 4 -1
	0 -1  0
	
	# Vertical Edge
	1 0 -1
	2 0 -2
	1 0 -1

**Hyperparameters** of a convolutional layer:

1. The number of filters (controls the **depth** of the output volume) 
2. The **stride**: how many units apart do we apply a filter spatially (this controls the spatial size of the output volume) 
3. The size w × h of the filters  

**Pooling**  gain robustness to the exact spatial location of features. 

#### 5. Ways to measure the size of a network

1. Number of units. This is important because the activations need to be stored in memory during training (i.e. backprop). 
2. Number of weights. This is important because the weights need to be stored in memory, and because the number of parameters determines the amount of overfitting. 
3. Number of connections. This is important because there are approximately 3 add-multiply operations per connection (1 for the forward pass, 2 for the backward pass). 

Most of the units and connections are in the convolution layers. 

Most of the weights are in the fully connected layers. 

#### 6. Visualization

we can understand what first-layer features are doing by visualizing the weight matrices. 

Visualize higher-level features by seeing what inputs activate them. 

One way to formalize: pick the images in the training set which activate a unit most strongly. 

Problems? 

1. Can’t tell what the unit is actually responding to in the image. 

2. We may read too much into the results, e.g. a unit may detect red, and the images that maximize its activation will all be stop signs.

Optimizing the Image 

- Optimize an image from scratch to increase a unit’s activation. 

- Can do gradient ascent on an image to maximize the activation of a given neuron. 

- Higher layers in the network often learn higher-level, more interpretable representations 


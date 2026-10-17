---
layout: post
title: "Machine Learning from Scratch 02: Multilayer Perceptrons"
date: 2026-10-17
description: "From a single perceptron to feedforward neural networks: hidden layers, activation functions, and the first steps towards training with gradient descent."
categories: [machine-learning, neural-networks, fundamentals]
image: /assets/images/ml02-feedforward-network.png
---

{% assign first_ml_post = site.posts | where_exp: "post", "post.title contains 'Machine Learning from Scratch 01'" | first %}

We left the previous discussion by introducing non-linearly separable data and the idea of a multilayer perceptron. So, what exactly is a multilayer perceptron?

## Multilayer perceptron

The basic idea is to add one or more layers of neurons between the inputs and the neuron, or neurons, that produce the final output.

![Comparison of network topologies and the decision regions they can represent](/assets/images/ml02-multilayer-perceptron.png)

This technique seems cool (and it is), but it also brings a few complications that deserve a closer look. First of all, how do we train the neurons in the “middle” layers? If you immediately thought of the supervised learning rule from the {% if first_ml_post %}[first post]({{ first_ml_post.url | relative_url }}){% else %}first post{% endif %}, remember that it requires a desired output for each neuron. For these intermediate neurons, we do not have one.

So, how does this model work?

## Feedforward neural networks

![A fully connected feedforward neural network with input, hidden, and output layers](/assets/images/ml02-feedforward-network.png)

This architecture is organized into layers:

- **Input layer:** passes the inputs to the following layer without performing any computation.
- **Hidden layer or layers:** the number of layers and the number of neurons in each are design choices, or hyperparameters to tune.
- **Output layer:** its size depends on how we represent the desired output of the problem.

Before continuing, let’s make a few assumptions:

- We use **fully connected layers**, meaning that every neuron in one layer is connected to every neuron in the next. If inputs are represented as row vectors and the bias is included as an extra constant input, the first weight matrix $W$ has shape $(I+1)\times J$, where $I$ is the number of input features and $J$ is the number of neurons in the first hidden layer.
- The output of each layer depends only on the previous layer.
- For the gradient-based training we are about to discuss, we initially consider differentiable activation functions. If you are rusty on the concept, you can review differentiability [here](https://en.wikipedia.org/wiki/Differentiable_function).

## Activation functions

Some possible activation functions are:

- **Linear:** returns its input unchanged. Its derivative is constant:

  $$
  h(a)=a,\qquad h'(a)=1.
  $$

- **Sigmoid:** an S-shaped function that maps any real input to a value between zero and one:

  $$
  h(a)=\frac{1}{1+e^{-a}},\qquad h'(a)=h(a)\bigl(1-h(a)\bigr).
  $$

  Its output approaches zero for large negative inputs and one for large positive inputs. Its range is $(0,1)$.

- **Tanh:** the hyperbolic tangent is also S-shaped, but its output is centred around zero:

  $$
  h(a)=\tanh(a),\qquad h'(a)=1-h(a)^2.
  $$

  Its output approaches $-1$ for large negative inputs and $+1$ for large positive inputs. Its range is $(-1,1)$.

![Linear, sigmoid, and tanh activation functions](/assets/images/ml02-activation-functions.png)

If you are new to this, you may be thinking: “The linear function is easy. Why are we considering more complicated functions?”

We do like linear functions. However, if we use **only linear activation functions**, the entire network collapses into a single affine transformation. For regression, it has no more expressive power than a linear regression model, despite using many more parameters.

So, how do we choose an activation function? It depends on the context.

Suppose we have a regression task whose output can take any value in $\mathbb{R}$. A linear activation is a natural choice for the output neuron. A sigmoid, for example, would restrict its output to $(0,1)$.

For classification, the choice also depends on the encoding. If we use $\Omega_0=-1$ and $\Omega_1=+1$, a tanh output can be used. If we use $\Omega_0=0$ and $\Omega_1=1$, a sigmoid output is a natural choice.

What if we have more than two classes? For a multiclass task with one correct class per example, we can use one output neuron per class and **one-hot encoding**. For three classes, one possible encoding is:

$$
\Omega_0=(0,0,1),\qquad
\Omega_1=(0,1,0),\qquad
\Omega_2=(1,0,0).
$$

The outputs can then be computed using **softmax**:

$$
y_k=\frac{\exp(z_k)}{\sum_{j=1}^{K}\exp(z_j)},
\qquad k=1,\ldots,K,
$$

where $z_k$ is the weighted sum entering output unit $k$, and $K$ is the number of classes. Softmax produces positive outputs that sum to one. The exponential amplifies differences between the scores, giving larger scores a larger share of the total. It provides a smooth alternative to selecting only the maximum, which is the idea behind the name. If you are interested, let me know and I will write a post about it.

We have discussed the output layer, but what about the hidden layers? For now, we will use tanh. ReLU is a common modern choice for hidden layers, but I will introduce it later in the series.

## Can we now approximate everything?

A result associated with [Kurt Hornik’s 1991 work](https://doi.org/10.1016/0893-6080(91)90009-T) tells us that a feedforward network with a single hidden layer, sufficiently many hidden units, a suitable continuous, bounded, nonconstant hidden activation function, and a linear output can approximate any continuous function arbitrarily closely on a compact input set.

This is a form of the [universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem). In particular, one hidden layer can be sufficient to represent complex classification boundaries.

Does this mean that a single hidden layer is the best architecture in every possible scenario? Obviously not. Knowing that suitable weights exist does not mean that a learning algorithm will find them. A shallow network may require a very large number of hidden units, and a large network can overfit. We will discuss generalization and overfitting later.

## How do we train neural networks?

Let’s take a step back to supervised learning.

Given training data

$$
\mathcal{D}=\{(\mathbf{x}_1,t_1),\ldots,(\mathbf{x}_N,t_N)\},
$$

we want to find weights such that the network’s prediction is close to the target:

$$
g(\mathbf{x}_n,\mathbf{w})\approx t_n.
$$

For a scalar regression output, we can express this goal by minimizing the sum of squared errors:

$$
E(\mathbf{w})=\sum_{n=1}^{N}\left[t_n-g(\mathbf{x}_n,\mathbf{w})\right]^2.
$$

![A regression line with squares illustrating the squared prediction errors](/assets/images/ml02-squared-error.png)

The goal in this illustration is to find the blue line that minimizes the total area of the red squares.

If you have spent a little time with calculus, you will recognize a standard first step: compute the partial derivatives and set them to zero:

$$
\nabla_{\mathbf{w}}E(\mathbf{w})=\mathbf{0}.
$$

This identifies stationary points, which still need to be checked to determine whether they are minima. For neural networks, solving this system in closed form is generally impractical. Instead, we use an iterative method: **gradient descent**.

## Gradient descent

The algorithm is fairly simple:

1. Initialize the weights randomly.
2. Repeatedly update them in the direction opposite to the gradient:

$$
\mathbf{w}^{(k+1)}=\mathbf{w}^{(k)}-\eta\nabla E\left(\mathbf{w}^{(k)}\right).
$$

Continue until a stopping criterion is reached.

![Gradient descent moving from an initial weight towards the minimum of a quadratic error function](/assets/images/ml02-gradient-descent-steps.png)

The reasoning is straightforward. In one dimension, a positive derivative means that the function increases as the weight increases, so we move in the opposite direction. In several dimensions, the negative gradient gives the direction of steepest local decrease. The learning rate $\eta$ scales the update and is another important hyperparameter to tune.

Unfortunately, the illustration above shows a nice quadratic error function. For a neural network, the error landscape can look more like this:

![A nonconvex error function with multiple local minima](/assets/images/ml02-nonconvex-error.png)

We still use the same squared-error objective,

$$
E(\mathbf{w})=\sum_{n=1}^{N}\left[t_n-g(\mathbf{x}_n,\mathbf{w})\right]^2,
$$

and perform gradient descent using

$$
\mathbf{w}^{(k+1)}=\mathbf{w}^{(k)}-\eta\nabla E\left(\mathbf{w}^{(k)}\right).
$$

If you are paying attention, you may now be wondering: “Since we start from random weights, what happens if we reach a local minimum rather than the global one?”

One modification that can improve the optimization process is **momentum**:

$$
\mathbf{w}^{(k+1)}
=\mathbf{w}^{(k)}
-\eta\nabla E\left(\mathbf{w}^{(k)}\right)
+\alpha\left(\mathbf{w}^{(k)}-\mathbf{w}^{(k-1)}\right).
$$

This introduces a form of inertia: we retain a component of the previous update direction. The coefficient $0\leq\alpha<1$ controls how much of that previous movement carries over. Momentum can help the optimization move through shallow regions and reduce oscillations, but it does not guarantee that we reach the global minimum.

We will continue from here in the third post.

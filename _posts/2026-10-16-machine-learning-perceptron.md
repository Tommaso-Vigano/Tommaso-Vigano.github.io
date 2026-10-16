---
layout: post
title: "Machine Learning from Scratch 01: The Perceptron"
date: 2026-10-16
description: "How does a model learn from examples? An introduction to learning paradigms, the perceptron, and the difference between learning OR and learning XOR."
categories: [machine-learning, neural-networks, fundamentals]
image: /assets/images/ml01-or-xor.png
---

A few days ago, I was discussing machine learning and artificial intelligence with a friend who studies economics. While trying to explain a few ideas, I realized that I was a little rusty on some of the fundamentals.

He suggested turning those explanations into a series of blog posts so that he could follow along so I decided to take up the challenge.

This is the first article in that series. 

You will need basic Python and some familiarity with vectors and elementary mathematics. Statistics and calculus will become more useful as the series progresses. If you already have a computer science background, some of these first explanations will be familiar.

We will start with a simple question: **how can a model learn a decision rule from examples?**

## Three ways to learn from data

Suppose we have a dataset of $N$ observations:

$$
\mathcal{D}=\{\mathbf{x}^{(1)},\mathbf{x}^{(2)},\ldots,\mathbf{x}^{(N)}\}.
$$

Each observation is a vector of $d$ features. For a house, those features might be its floor area, number of rooms, distance from the city centre...

What we ask a model to learn depends on the information available during training. Three common learning paradigms are supervised learning, unsupervised learning, and reinforcement learning.

### Supervised learning

In supervised learning, each input comes with a target, or desired output:

$$
\mathcal{D}=\{(\mathbf{x}^{(1)},t^{(1)}),\ldots,(\mathbf{x}^{(N)},t^{(N)})\}.
$$

The model learns a relationship between inputs and targets so that it can predict an output for a new input.

If the target is a category, such as recognizing if an email is spam, the task is **classification**. If it is a numerical quantity, such as a house price, the task is **regression**.

### Unsupervised learning

In unsupervised learning, we have observations but no target labels. We look for structure in the data.

For example, a clustering algorithm might group customers with similar purchasing habits. We do not give it the correct customer group for each observation; the groups emerge from the chosen algorithm and its notion of similarity.

Those groups still need interpretation. A cluster is not automatically a meaningful real-world category.

![Classification with known labels compared with clustering without target labels](/assets/images/ml01-learning-paradigms.png)


### Reinforcement learning

In reinforcement learning, an **agent** interacts with an environment. It observes a situation, chooses an action, and receives feedback through rewards and observations.

The objective is to learn a policy for choosing actions that maximizes expected cumulative reward. An action may have consequences that only become apparent much later, so maximizing the next reward is not always the right strategy as you could think.

A game-playing agent, for example, may sacrifice a piece now to improve its chances of winning later. 
Personally, I think that this is the most interesting paradigm, even it's the more complicated.

## Where deep learning fits

Machine learning is part of artificial intelligence. Deep learning is a family of machine learning methods that uses multiple layers of learned transformations to build representations of the input.

To understand why representations matter, imagine trying to separate points arranged in two concentric rings. A straight line cannot separate the inner ring from the outer one in the original coordinates.

However, consider the feature:

$$
\phi(\mathbf{x})=x_1^2+x_2^2.
$$

This is the squared distance from the origin. In this representation, a threshold can distinguish the rings.

![Concentric classes in the original coordinates and their separation using squared distance from the origin](/assets/images/ml01-feature-representation.png)

*The same observations can become easier to classify after a change of representation.*

Here, we chose the transformation ourselves. A neural network can instead learn useful transformations from data. In a supervised classification model, the representations and the final classifier can be trained together.

This is often part of **end-to-end learning**, where a system is trained from its inputs to its final outputs. However, end-to-end learning and deep learning are not interchangeable terms: the former describes how a system is trained, while the latter concerns the depth of its learned transformations.

To understand how neural networks make and learn decisions, we will start with a single and basic computational unit: the perceptron.

## The perceptron model

The perceptron is inspired by biological neurons: it combines incoming signals and produces an output according to a threshold. Clearly it is a mathematical abstraction, not a detailed simulation of how a real neuron works.

Given $d$ input features, it first computes a weighted sum:

$$
z=b+\sum_{i=1}^{d}w_i x_i.
$$

Each weight $w_i$ determines how its input contributes to the sum. The **bias** $b$ shifts the threshold at which the output changes.

An activation function then converts the sum into a prediction:

$$
\hat{y}=h(z).
$$

![Inputs and a constant bias input feeding a weighted sum and sign activation in a perceptron](/assets/images/ml01-perceptron.png)

For this article, we will use the sign function:

$$
h(z)=
\begin{cases}
+1 & \text{if } z>0,\\
0 & \text{if } z=0,\\
-1 & \text{if } z<0.
\end{cases}
$$


We can absorb the bias into the weight vector by introducing a constant input $x_0=1$ and setting $w_0=b$:

$$
\tilde{\mathbf{x}}=(1,x_1,\ldots,x_d),
\qquad
\tilde{\mathbf{w}}=(w_0,w_1,\ldots,w_d).
$$

The prediction becomes:

$$
\hat{y}=h\left(\tilde{\mathbf{w}}^\top\tilde{\mathbf{x}}\right).
$$

There are $d$ measured inputs and $d+1$ trainable parameters, including the bias. If a text instead writes the sum minus a threshold $\theta$, the equivalent bias is $b=-\theta$.

## A perceptron that implements OR

Before looking how does the model learn the weights, let’s check that a perceptron can implement a familiar Boolean function.

The OR operator returns true whenever at least one input is true:

| Input 1 | Input 2 | OR |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

Using inputs encoded as zero and one, choose:

$$
z=-\frac{1}{2}+x_1+x_2.
$$

The score is negative only for $(0,0)$ and positive for the other three inputs. Interpreting a negative prediction as false and a positive prediction as true gives OR.

But we selected these weights following the literature. How could a model find suitable weights from the examples?

## Hebbian Learning Rule

We will use the **Hebbian learning rule**. For each training example, compute the score and update the weights if the prediction is wrong by using this rule:

$$
\begin{aligned}
w_i^{(k+1)} &= w_i^k + \Delta(w_i^k) \\
\Delta(w_i^k) &= \eta x_i^k t^k
\end{aligned}
$$

Otherwise, leave the weights unchanged.

Here, $\eta>0$ is the **learning rate**, which controls the size of an update; $t$ is the target i.e. the desired output; and $k$ counts parameter updates. 

## Learning OR step by step

For training, we will encode false as $-1$ and true as $+1$, for both inputs and targets. This changes the numerical representation, while preserving the OR relationship:

| $x_1$ | $x_2$ | Target $t$ |
| --- | --- | --- |
| $-1$ | $-1$ | $-1$ |
| $-1$ | $+1$ | $+1$ |
| $+1$ | $-1$ | $+1$ |
| $+1$ | $+1$ | $+1$ |

We will visit the examples in this order, initialize all weights to zero, and use a learning rate of $0.5$:

$$
\tilde{\mathbf{w}}^{(0)}=(0,0,0),\qquad \eta=0.5.
$$

The order of the weights is always $(w_0,w_1,w_2)$, and each input is $(1,x_1,x_2)$.

### First epoch

An **epoch** is one complete pass through the training dataset.

For the first example, the input is $(1,-1,-1)$ and the target is $-1$. With zero weights, the score is zero, so an update is required:

$$
\Delta\tilde{\mathbf{w}}
=0.5(-1)(1,-1,-1)
=(-0.5,0.5,0.5).
$$

The new weights are $(-0.5,0.5,0.5)$. They now give this example a score of $-1.5$, which has the correct sign.

For the second example, $(1,-1,+1)$, the target is $+1$. Its score is:

$$
z=-0.5+0.5(-1)+0.5(+1)=-0.5.
$$

The prediction is negative, so we update:

$$
\tilde{\mathbf{w}}^{(2)}
=(-0.5,0.5,0.5)+0.5(+1)(1,-1,+1)
=(0,0,1).
$$

For the third example, $(1,+1,-1)$, the target is also $+1$. The current score is $-1$, so another update gives:

$$
\tilde{\mathbf{w}}^{(3)}
=(0,0,1)+0.5(+1)(1,+1,-1)
=(0.5,0.5,0.5).
$$

The fourth example, $(1,+1,+1)$, then has score $1.5$. Its target is positive, so no update is needed.

At the end of the first epoch, the weights are:

$$
\tilde{\mathbf{w}}=(0.5,0.5,0.5).
$$

### Second epoch

An update that fixes one example can change predictions for earlier examples. We therefore make another complete pass using the current weights:

| Inputs $(x_1,x_2)$ | Score $0.5+0.5x_1+0.5x_2$ | Prediction | Target | Update |
| --- | --- | --- | --- | --- |
| $(-1,-1)$ | $0.5-0.5-0.5=-0.5$ | $-1$ | $-1$ | No |
| $(-1,+1)$ | $0.5-0.5+0.5=0.5$ | $+1$ | $+1$ | No |
| $(+1,-1)$ | $0.5+0.5-0.5=0.5$ | $+1$ | $+1$ | No |
| $(+1,+1)$ | $0.5+0.5+0.5=1.5$ | $+1$ | $+1$ | No |

Every example is correct and no weights change. Training stops.


## Visualizing the perceptron

We have seen that a perceptron computes a weighted sum of its inputs and returns its sign. We can also interpret this operation geometrically: the perceptron is a **linear classifier**, whose decision boundary is the hyperplane

$$
w_0 + w_1 x_1 + \cdots + w_I x_I = 0.
$$

Points on one side produce a positive weighted sum; points on the other side produce a negative one.

With two input features, $I=2$, the boundary becomes

$$
w_0 + w_1 x_1 + w_2 x_2 = 0.
$$

Assuming $w_2 \neq 0$, we can rearrange it step by step:

$$
\begin{aligned}
w_2 x_2 &= -w_0 - w_1 x_1 \\
x_2 &= -\frac{w_0}{w_2} - \frac{w_1}{w_2} x_1.
\end{aligned}
$$

This is the equation of a straight line. The weights determine its slope and position, while the sign of the weighted sum determines the predicted class.

This is exactly how the perceptron implements Boolean operators such as OR: it finds a line that separates the inputs associated with a negative target from those associated with a positive target.

We can now state which Boolean functions a single perceptron can learn: **those whose positive and negative examples are linearly separable in the given input space**.

XOR is an example of a Boolean function that does not satisfy this condition. It returns true when exactly one input is true. Using our bipolar notation, its positive examples are $(-1,+1)$ and $(+1,-1)$, while its negative examples are $(-1,-1)$ and $(+1,+1)$.

![OR examples separated by a straight line alongside XOR examples that cannot be separated by one line](/assets/images/ml01-or-xor.png)

*OR is linearly separable. For XOR, the classes occupy opposite corners, so no single straight line can separate them.*

In this situation, a single perceptron cannot solve the problem using the original inputs. We need a different representation in which the classes become linearly separable.

Learning such representations through additional layers is the idea we will explore with **multilayer perceptrons** in the next article.

---

## Sources and links

- [Bernhard Mehlig — Machine Learning with Neural Networks](https://arxiv.org/abs/1901.05639)
- [Stanford CS229 — Lecture notes on supervised learning and the perceptron](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)
- [Cornell CS4780 — The perceptron and its convergence theorem](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote03.html)
- [Goodfellow, Bengio and Courville — Deep Learning, Chapter 1](https://www.deeplearningbook.org/contents/intro.html)

The diagrams in this article were created for these explanations using illustrative data.

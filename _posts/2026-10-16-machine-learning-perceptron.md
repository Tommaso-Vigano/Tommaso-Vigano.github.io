---
layout: post
title: "Machine Learning from Scratch 01: The Perceptron"
date: 2026-10-16
description: "How does a model learn from examples? An introduction to learning paradigms, the perceptron, and the difference between learning OR and learning XOR."
categories: [machine-learning, neural-networks, fundamentals]
image: /assets/images/ml01-or-xor.png
---

A few days ago, I was discussing machine learning and artificial intelligence with a friend who studies economics. While trying to explain a few ideas, I realized that I was a little rusty on some of the fundamentals myself.

He suggested turning those explanations into a series of blog posts so that he could follow along. I decided to take up the challenge.

This is the first article in that series. The goal is to revisit the foundations of machine learning and neural networks, explain the mathematics, and connect it to small Python examples.

You will need basic Python and some familiarity with vectors and elementary mathematics. Statistics and calculus will become more useful as the series progresses. If you already have a computer science background, some of these first explanations will be familiar.

We will start with a simple question: **how can a model learn a decision rule from examples?**

## Three ways to learn from data

Suppose we have a dataset of $N$ observations:

$$
\mathcal{D}=\{\mathbf{x}^{(1)},\mathbf{x}^{(2)},\ldots,\mathbf{x}^{(N)}\}.
$$

Each observation is a vector of $d$ features. For a house, those features might include its floor area, number of rooms, and distance from the city centre.

What we ask a model to learn depends on the information available during training. Three common learning paradigms are supervised learning, unsupervised learning, and reinforcement learning.

### Supervised learning

In supervised learning, each input comes with a target, or desired output:

$$
\mathcal{D}=\{(\mathbf{x}^{(1)},t^{(1)}),\ldots,(\mathbf{x}^{(N)},t^{(N)})\}.
$$

The model learns a relationship between inputs and targets so that it can predict an output for a new input.

If the target is a category, such as whether an email is spam, the task is **classification**. If it is a numerical quantity, such as a house price, the task is **regression**.

Doing well on examples seen during training is only part of the goal. We also want the model to work on new examples. This ability is called **generalization**.

### Unsupervised learning

In unsupervised learning, we have observations but no target labels. We look for structure in the data.

For example, a clustering algorithm might group customers with similar purchasing habits. We do not give it the correct customer group for each observation; the groups emerge from the chosen algorithm and its notion of similarity.

Those groups still need interpretation. A cluster is not automatically a meaningful real-world category.

![Classification with known labels compared with clustering without target labels](/assets/images/ml01-learning-paradigms.png)

*Illustrative data: classification uses supplied labels, while clustering searches for groups without them.*

### Reinforcement learning

In reinforcement learning, an **agent** interacts with an environment. It observes a situation, chooses an action, and receives feedback through rewards and subsequent observations.

The objective is to learn a policy for choosing actions that maximizes expected cumulative reward. An action may have consequences that only become apparent much later, so maximizing the next reward is not always the right strategy.

A game-playing agent, for example, may sacrifice a piece now to improve its chances of winning later. Personally, I find this interaction between decisions and delayed consequences particularly interesting.

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

To understand how neural networks make and learn decisions, we will start with a single computational unit: the perceptron.

## The perceptron model

The perceptron is loosely inspired by biological neurons: it combines incoming signals and produces an output according to a threshold. It is a mathematical abstraction, not a detailed simulation of how a neuron works.

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

Our target classes will be $-1$ and $+1$. The value zero represents an undecided point exactly on the decision boundary, and we will treat it as requiring a training update. Other tie conventions are possible, but we need to choose one consistently.

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

Before learning the weights, let’s check that a perceptron can implement a familiar Boolean function.

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

But we selected these weights by hand. How could a model find suitable weights from the examples?

## Learning the weights

We will use the **perceptron learning rule**. For each training example, compute the score and update the weights if the prediction is wrong or the score is zero:

$$
\tilde{\mathbf{w}}^{(k+1)}=
\tilde{\mathbf{w}}^{(k)}+
\eta\,t\,\tilde{\mathbf{x}}
\quad\text{when }t z\leq 0.
$$

Otherwise, leave the weights unchanged.

Here, $\eta>0$ is the **learning rate**, which controls the size of an update; $t$ is the target; and $k$ counts parameter updates. Because $\tilde{\mathbf{x}}$ includes the constant input, the same rule also updates the bias.

For a positive example, the update adds a multiple of the input vector. For a negative example, it subtracts one. In either case, the update moves the example’s score toward the desired sign.

This rule has a correlation-based form associated with Hebbian ideas, but the instruction to update only on errors or boundary cases is important. The procedure here is specifically **perceptron learning**, rather than the basic Hebbian rule.

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

The order of the weights is always $(w_0,w_1,w_2)$, and each augmented input is $(1,x_1,x_2)$.

### First epoch

An **epoch** is one complete pass through the training dataset.

For the first example, the augmented input is $(1,-1,-1)$ and the target is $-1$. With zero weights, the score is zero, so an update is required:

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

## The same example in Python

The following script reproduces both epochs. The first column of `X` is the constant bias input:

```python
import numpy as np

X = np.array([
    [1, -1, -1],
    [1, -1,  1],
    [1,  1, -1],
    [1,  1,  1],
], dtype=float)

targets = np.array([-1, 1, 1, 1])
weights = np.zeros(X.shape[1])
learning_rate = 0.5
max_epochs = 100

for epoch in range(1, max_epochs + 1):
    updates = 0

    for x, target in zip(X, targets):
        score = weights @ x
        if target * score <= 0:
            weights += learning_rate * target * x
            updates += 1

    print(f"Epoch {epoch}: {updates} updates; weights = {weights}")

    if updates == 0:
        break
else:
    raise RuntimeError("Training did not converge within max_epochs.")

predictions = np.sign(X @ weights).astype(int)
print("Predictions:", predictions)
```

The output is:

```text
Epoch 1: 3 updates; weights = [0.5 0.5 0.5]
Epoch 2: 0 updates; weights = [0.5 0.5 0.5]
Predictions: [-1  1  1  1]
```

The epoch limit prevents an endless training loop on a dataset the perceptron cannot separate. Reaching that limit alone does not prove that separation is impossible; a separable problem may simply need more iterations.

## What the perceptron learns geometrically

The output changes where the weighted sum is zero. In two dimensions, the **decision boundary** is:

$$
w_0+w_1x_1+w_2x_2=0.
$$

When $w_2\neq 0$, we can rearrange this as:

$$
x_2=-\frac{w_0}{w_2}-\frac{w_1}{w_2}x_1.
$$

This is a straight line. If $w_2=0$ and $w_1\neq 0$, the boundary is a vertical line instead. With more input dimensions, the equivalent boundary is a hyperplane.

For the OR weights learned above, the boundary is:

$$
0.5+0.5x_1+0.5x_2=0,
\qquad\text{or}\qquad
x_2=-1-x_1.
$$

It separates the single negative example from the three positive examples.

![OR examples separated by a straight line alongside XOR examples that cannot be separated by one line](/assets/images/ml01-or-xor.png)

*Both plots use bipolar inputs and targets. The line on the left corresponds to the OR weights learned in the example.*

Different initial weights or a different order of examples can produce another valid separator. Starting from $(0,2,1)$ with the same learning rate and sample order, for example, leads to $(0.5,1.5,1.5)$. These weights also classify all four OR examples correctly, but define a different line.

Learning therefore does not necessarily recover a unique set of parameters.

## Why XOR is different

The **perceptron convergence theorem** guarantees that the standard algorithm, with a fixed positive learning rate and repeated passes through a finite, linearly separable training set, finds a separator after finitely many updates.

The condition matters: the classes must be separable by a line, or a hyperplane in higher dimensions, in the representation supplied to the model.

OR satisfies that condition. XOR does not.

XOR is true when exactly one input is true. In bipolar notation, its positive examples are $(-1,+1)$ and $(+1,-1)$, while its negative examples are $(-1,-1)$ and $(+1,+1)$.

As the right-hand plot shows, the classes occupy opposite corners. No single straight line can separate them perfectly. A single perceptron receiving only these two input features cannot represent XOR, regardless of how long we train it.

This limitation concerns the model and its input representation. It does not mean that XOR is impossible for neural networks. Nonlinear transformations can make the classes separable in another representation; multilayer networks can learn such transformations.

That is where we will pick up in the next article: how combining computational units into a multilayer perceptron changes the decisions a network can represent.

---

## Sources and links

- [Bernhard Mehlig — Machine Learning with Neural Networks](https://arxiv.org/abs/1901.05639)
- [Stanford CS229 — Lecture notes on supervised learning and the perceptron](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)
- [Cornell CS4780 — The perceptron and its convergence theorem](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote03.html)
- [Goodfellow, Bengio and Courville — Deep Learning, Chapter 1](https://www.deeplearningbook.org/contents/intro.html)

The diagrams in this article were created for these explanations using illustrative data.

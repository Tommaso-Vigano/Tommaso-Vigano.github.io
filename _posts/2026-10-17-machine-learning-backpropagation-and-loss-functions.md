---
layout: post
title: "Machine Learning from Scratch 03: Backpropagation and Loss Functions"
date: 2026-10-17
description: "From computing gradients with backpropagation to understanding why regression and classification use different loss functions."
categories: [machine-learning, neural-networks, fundamentals]
image: /assets/images/ml03-network.png
---

Let’s start from where we left off, with an architecture like this:

![A neural network with one hidden layer and a single output](/assets/images/ml03-network.png)

We know that the error can be written as:

$$
E(\mathbf{w})=\sum_{n=1}^{N}\left[t_n-g_1(\mathbf{x}_n,\mathbf{w})\right]^2.
$$

For a weight $w_{ji}^{(1)}$ connecting input $i$ to hidden unit $j$, its derivative is:

$$
\begin{aligned}
\frac{\partial E}{\partial w_{ji}^{(1)}}
&=-2\sum_{n=1}^{N}\left[t_n-g_1(\mathbf{x}_n,\mathbf{w})\right]\\
&\qquad\cdot\phi'(b_n)\,w_{1j}^{(2)}\,h_j'(a_{j,n})\,x_{i,n}.
\end{aligned}
$$

Here, $a_{j,n}$ is the weighted sum entering hidden unit $j$, $b_n$ is the weighted sum entering the output unit, and $\phi$ is the output activation:

$$
\begin{aligned}
a_{j,n}&=\sum_{i=0}^{I}w_{ji}^{(1)}x_{i,n},\\
b_n&=\sum_{j=0}^{J}w_{1j}^{(2)}h_j(a_{j,n}),\\
g_1(\mathbf{x}_n,\mathbf{w})&=\phi(b_n).
\end{aligned}
$$

The index $0$ accounts for the biases: $x_{0,n}=1$ and the hidden layer’s extra constant output is $h_0=1$.

Unlike the online learning rule we discussed earlier, we are now considering an error computed over the entire dataset. **Online** means that we update the weights after each sample, not that we update only one weight at a time. In either case, operations on many weights can be computed in parallel.

This is cool, but it can become impractical. Let me be more precise: when we work with large neural networks, we often use a GPU. Moving data between the computer’s main memory and the GPU’s memory takes time, so we want to organize those transfers efficiently.

GPU memory is limited, and it must also hold the model and the intermediate values needed for training. The entire dataset may not fit at once.

In **batch gradient descent**, each update uses all $N$ training samples. If we write the error for one sample as $E_n(\mathbf{w})=[t_n-g_1(\mathbf{x}_n,\mathbf{w})]^2$, the gradient of the average error is:

$$
\nabla_{\mathbf{w}}\left[\frac{E(\mathbf{w})}{N}\right]
=\frac{1}{N}\sum_{n=1}^{N}\nabla_{\mathbf{w}}E_n(\mathbf{w}).
$$

Using the average instead of the sum only rescales the gradient. Also, full-batch training does not require all the data to be on the GPU simultaneously: we can accumulate the gradient over smaller chunks before updating the weights.

What if we update after just one sample? Well, I hope at least one sample fits! This gives us **stochastic gradient descent**, or **SGD**, the equivalent of online learning:

$$
\nabla_{\mathbf{w}}\left[\frac{E(\mathbf{w})}{N}\right]
\approx\nabla_{\mathbf{w}}E_n(\mathbf{w}).
$$

Here, sample $n$ is selected randomly. Its gradient estimates the gradient of the average error.

And what if we use only part of the dataset for each update? That is **mini-batch gradient descent**, which is common practice. For a mini-batch $\mathcal{B}$ containing $M$ samples:

$$
\nabla_{\mathbf{w}}\left[\frac{E(\mathbf{w})}{N}\right]
\approx\frac{1}{M}\sum_{n\in\mathcal{B}}\nabla_{\mathbf{w}}E_n(\mathbf{w}).
$$

Introducing these different versions of gradient descent means that we need to distinguish **iterations** from **epochs**.

An iteration is one update of the weights. An epoch is one complete pass through the training data. In SGD, visiting each of the $N$ samples once requires $N$ iterations per epoch. In batch gradient descent, one iteration is one epoch. With mini-batches of size $M$, an epoch requires $\lceil N/M\rceil$ iterations if we keep the final, smaller batch.

Now that these concepts are clear, I want to make one thing explicit: **you will not compute all these derivatives by hand**.

The gradients can be computed using **automatic differentiation**, or **AutoDiff**. Let’s explain this concept, starting from the basics.

Do you remember how to differentiate a composite function? If not, let me refresh your memory.

Let $x$ be a real number and let $f:\mathbb{R}\to\mathbb{R}$ and $g:\mathbb{R}\to\mathbb{R}$. Consider the composite function $z=f(g(x))=f(y)$, where $y=g(x)$. Applying the chain rule gives:

$$
\frac{dz}{dx}
=\frac{dz}{dy}\frac{dy}{dx}
=f'(y)g'(x)
=f'(g(x))g'(x).
$$

This is the same reasoning we apply in **backpropagation**. For one sample, the derivative with respect to a hidden-layer weight follows the chain:

$$
\frac{\partial E_n}{\partial w_{ji}^{(1)}}
=\frac{\partial E_n}{\partial g_1}
\frac{\partial g_1}{\partial b_n}
\frac{\partial b_n}{\partial h_j}
\frac{\partial h_j}{\partial a_{j,n}}
\frac{\partial a_{j,n}}{\partial w_{ji}^{(1)}}.
$$

Multiplying these terms gives the contribution of that sample to the derivative we wrote at the beginning.

Computing the intermediate values—such as the hidden activations $h_j(a_{j,n})$, their weighted contributions, and the final prediction $g_1(\mathbf{x}_n,\mathbf{w})$—is called the **forward pass**.

Propagating derivatives back through these operations using the chain rule is called the **backward pass**. When a value contributes through several paths, we add their derivative contributions.

![A forward pass computes the prediction; a backward pass propagates derivatives towards the weights](/assets/images/ml03-backpropagation.png)

Backpropagation is the application of reverse-mode automatic differentiation to the network. Gradient descent then uses the resulting gradients to update the weights.

PyTorch, the library we will use in this series, provides automatic differentiation for this purpose.

If there is one thing I love doing, it is questioning why we use the tools we use. So we have to ask: is it always a good choice to use this error function?

$$
E(\mathbf{w})=\sum_{n=1}^{N}\left[t_n-g_1(\mathbf{x}_n,\mathbf{w})\right]^2.
$$

Clearly, if I am bringing this up, the answer is no. But let’s understand why.

## Understanding maximum likelihood estimation

Let’s observe independent and identically distributed (**i.i.d.**) samples from a Gaussian distribution with known variance $\sigma^2$:

$$
x_1,x_2,\ldots,x_N\sim\mathcal{N}(\mu,\sigma^2).
$$

Its probability density is:

$$
p(x\mid\mu,\sigma^2)
=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right].
$$

![Three Gaussian distributions with different means compared with the observed samples](/assets/images/ml03-maximum-likelihood.png)

**Maximum likelihood** means choosing the parameters that maximize the likelihood of the observed data. For a continuous distribution, we use probability densities rather than the probability of observing an exact value.

If we have to choose between the three models in the picture, the purple one best matches our data.

Notice that we are using a Gaussian distribution here because it is convenient: it has only two parameters and is easy to work with. Nobody forces us to choose it over another distribution. The important point is to choose the parameters that maximize the likelihood **within the model we have selected**.

We need a little statistical estimation here. Maximum likelihood itself does not require Bayesian inference or a prior distribution.

Let $\boldsymbol{\theta}=(\theta_1,\theta_2,\ldots,\theta_p)^T$ be a vector of parameters. To find its maximum likelihood estimate:

1. Write the likelihood $L(\boldsymbol{\theta})=p(\mathcal{D}\mid\boldsymbol{\theta})$ for the observed data.
2. Optionally, take its logarithm: $\ell(\boldsymbol{\theta})=\log L(\boldsymbol{\theta})$.
3. Compute the derivatives of $L$ or $\ell$ with respect to the parameters.
4. For an interior solution, solve $\partial L/\partial\theta_i=0$ or $\partial\ell/\partial\theta_i=0$ for all $i$.
5. Check that the candidate is a maximum, considering any parameter constraints or boundary solutions as well.

Why use the **log-likelihood**? It makes things simpler: products become sums, and differentiating sums is much easier. The logarithm is also strictly increasing, so it does not change where the maximum occurs.

Depending on the problem, we can use analytical techniques, methods for constrained optimization such as Lagrange multipliers, or numerical techniques such as gradient ascent on the log-likelihood. Equivalently, we can use gradient descent on the negative log-likelihood.

Let’s work through the Gaussian example. Since the samples are independent, the likelihood is the product of their densities:

$$
\begin{aligned}
L(\mu)
&=\prod_{n=1}^{N}p(x_n\mid\mu,\sigma^2)\\
&=\prod_{n=1}^{N}\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(x_n-\mu)^2}{2\sigma^2}\right].
\end{aligned}
$$

Taking the logarithm:

$$
\begin{aligned}
\ell(\mu)
&=\sum_{n=1}^{N}\log p(x_n\mid\mu,\sigma^2)\\
&=-\frac{N}{2}\log(2\pi\sigma^2)
-\frac{1}{2\sigma^2}\sum_{n=1}^{N}(x_n-\mu)^2.
\end{aligned}
$$

Now differentiate with respect to $\mu$:

$$
\frac{d\ell}{d\mu}
=\frac{1}{\sigma^2}\sum_{n=1}^{N}(x_n-\mu).
$$

Setting this equal to zero gives:

$$
\begin{aligned}
\sum_{n=1}^{N}(x_n-\mu)&=0,\\
\sum_{n=1}^{N}x_n&=N\mu,\\
\mu_{\mathrm{MLE}}&=\frac{1}{N}\sum_{n=1}^{N}x_n.
\end{aligned}
$$

So the maximum likelihood estimate of the mean is simply the sample mean. We can also check that it is a maximum:

$$
\frac{d^2\ell}{d\mu^2}=-\frac{N}{\sigma^2}<0.
$$

Let’s apply the same reasoning to a neural network used for regression. Suppose the targets follow:

$$
t_n=g_1(\mathbf{x}_n,\mathbf{w})+\epsilon_n,
\qquad \epsilon_n\sim\mathcal{N}(0,\sigma^2),
$$

with independent noise terms and the same variance for every observation. Equivalently:

$$
t_n\mid\mathbf{x}_n,\mathbf{w}
\sim\mathcal{N}\left(g_1(\mathbf{x}_n,\mathbf{w}),\sigma^2\right).
$$

The likelihood of the targets, given the inputs, is:

$$
L(\mathbf{w})
=\prod_{n=1}^{N}\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{[t_n-g_1(\mathbf{x}_n,\mathbf{w})]^2}{2\sigma^2}\right].
$$

Taking the logarithm gives:

$$
\ell(\mathbf{w})
=-\frac{N}{2}\log(2\pi\sigma^2)
-\frac{1}{2\sigma^2}\sum_{n=1}^{N}[t_n-g_1(\mathbf{x}_n,\mathbf{w})]^2.
$$

For fixed $\sigma^2$, the first term is constant with respect to the weights. Maximizing the log-likelihood therefore means minimizing the sum of squared errors:

$$
\underset{\mathbf{w}}{\operatorname{arg\,max}}\;\ell(\mathbf{w})
=\underset{\mathbf{w}}{\operatorname{arg\,min}}\;
\sum_{n=1}^{N}[t_n-g_1(\mathbf{x}_n,\mathbf{w})]^2.
$$

It turns out that the sum of squared errors makes perfect sense under these assumptions. Can we still use it when the assumptions do not hold? Yes, but it need not correspond to maximum likelihood under the actual noise model. It can still work, even if that model is not a perfect fit.

Here is a case in which the constant-variance assumption is questionable:

![Regression observations whose spread around the fitted line increases from left to right](/assets/images/ml03-changing-noise.png)

The noise clearly increases from left to right. What happens if we use the sum of squared errors? Large residuals count more because we square them, so the noisier observations can have a stronger influence on the fit.

How can we account for that? If observation $n$ has a known noise variance $\sigma_n^2$, the Gaussian likelihood leads to **weighted least squares**:

$$
E(\mathbf{w})
=\sum_{n=1}^{N}\frac{[t_n-g_1(\mathbf{x}_n,\mathbf{w})]^2}{\sigma_n^2}.
$$

An observation with greater noise variance receives less weight. This follows the same likelihood reasoning, with a different variance for each observation.

I’ll be honest with you: in regression, squared error is often a useful starting point, even when the assumptions are not perfectly satisfied.

Where does the Gaussian model become less appropriate? Classification is a good example. A target taking values in $\{0,1\}$ is not naturally described as a nonlinear function plus Gaussian noise.

Squared error can still be used for classification, but let’s follow the likelihood approach and choose a distribution that matches a binary target: the **Bernoulli distribution**.

The network now predicts the probability of class $1$:

$$
p_n=g_1(\mathbf{x}_n,\mathbf{w})
=P(t_n=1\mid\mathbf{x}_n,\mathbf{w}),
$$

typically using a sigmoid output so that $0<p_n<1$. We model:

$$
t_n\mid\mathbf{x}_n,\mathbf{w}\sim\operatorname{Bernoulli}(p_n).
$$

For a target $t_n\in\{0,1\}$, its probability is:

$$
P(t_n\mid\mathbf{x}_n,\mathbf{w})=p_n^{t_n}(1-p_n)^{1-t_n}.
$$

Assuming independent observations, the likelihood is:

$$
L(\mathbf{w})=\prod_{n=1}^{N}p_n^{t_n}(1-p_n)^{1-t_n}.
$$

The log-likelihood becomes:

$$
\ell(\mathbf{w})
=\sum_{n=1}^{N}\left[t_n\log p_n+(1-t_n)\log(1-p_n)\right].
$$

Maximizing this expression is equivalent to minimizing its negative. This gives us **binary cross-entropy**:

$$
E(\mathbf{w})
=-\sum_{n=1}^{N}\left[t_n\log p_n+(1-t_n)\log(1-p_n)\right].
$$

What happens when we have more than two classes?

If the $K$ classes are **mutually exclusive**, each sample belongs to exactly one class. We represent its target with a one-hot vector: $t_{n,k}=1$ for the correct class and $0$ for the others.

As we saw in the previous post, a **softmax** output transforms the network’s scores $z_{n,k}$ into class probabilities:

$$
p_{n,k}
=\frac{e^{z_{n,k}}}{\sum_{j=1}^{K}e^{z_{n,j}}}.
$$

These probabilities sum to one. Using a categorical distribution and applying the same likelihood reasoning gives **categorical cross-entropy**:

$$
E(\mathbf{w})
=-\sum_{n=1}^{N}\sum_{k=1}^{K}t_{n,k}\log p_{n,k}.
$$

Because the target is one-hot, only the predicted probability of the correct class contributes to each sample’s loss.

And what if the classes are not mutually exclusive? In **multilabel classification**, a sample can have several labels at once.

We then use one sigmoid output per label, rather than softmax. Each output predicts whether its label is present, and the probabilities do not need to sum to one. Modeling the labels as conditionally independent Bernoulli variables gives a binary cross-entropy term for each label:

$$
\begin{aligned}
E(\mathbf{w})=-\sum_{n=1}^{N}\sum_{k=1}^{K}
\big[&t_{n,k}\log p_{n,k}\\
&+(1-t_{n,k})\log(1-p_{n,k})\big].
\end{aligned}
$$

Unfortunately, I studied from some rather old books, and I still cannot quite let go of the term *error function*. In modern literature—and in Python libraries—you will often see it called a **loss function** instead.

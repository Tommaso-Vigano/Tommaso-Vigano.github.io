---
layout: post
title: "Machine Learning from Scratch 05: Dropout and Practical Tips for Training Neural Networks"
date: 2026-10-21
description: "Dropout, activation functions, Xavier and He initialization, and batch normalization: how these choices affect the training of neural networks."
categories: [machine-learning, neural-networks, fundamentals]
image: /assets/images/ml05-dropout.png
---

## Dropout

So far, we have not considered a technique that limits overfitting by temporarily changing which neurons participate in the network during training. This is the idea behind **dropout**.

![A neural network with some units temporarily dropped during training](/assets/images/ml05-dropout.png)

In a large, complex model, neurons can become too dependent on particular combinations of other neurons. Dropout discourages this dependence by randomly setting some neuron outputs to zero during training. The picture shows dropout in several layers, but we can apply it only to selected layers.

Mathematically, we generate a random **mask** for layer $l$:

$$
\begin{aligned}
\mathbf{m}^{(l)}&=\left[m_1^{(l)},\ldots,m_{J_l}^{(l)}\right]^T,\\
m_j^{(l)}&\sim\operatorname{Bernoulli}\left(p_j^{(l)}\right).
\end{aligned}
$$

Here, $J_l$ is the number of units in the layer, and $p_j^{(l)}$ is the probability of **keeping** neuron $j$ active. Its probability of being dropped is $1-p_j^{(l)}$. The superscript $l$ identifies the layer, not the training iteration.

How do we switch off an output? With the **Hadamard product**, written $\odot$: an element-by-element multiplication. Multiplying by zero removes that neuron's contribution.

If all units in the layer use the same keep probability $0<p\leq1$, a common implementation is:

$$
\begin{aligned}
\mathbf{a}^{(l)}
&=f^{(l)}\left(W^{(l)}\widetilde{\mathbf{a}}^{(l-1)}+\mathbf{b}^{(l)}\right),\\
\widetilde{\mathbf{a}}^{(l)}
&=\frac{\mathbf{m}^{(l)}\odot\mathbf{a}^{(l)}}{p}.
\end{aligned}
$$

We first compute the activations, then apply the mask. A dropped activation becomes zero; a retained activation is divided by $p$. This is called **inverted dropout**. The scaling keeps the expected activation unchanged. With different keep probabilities per neuron, we divide each retained activation by its own $p_j^{(l)}$ instead.

So what actually happens, and why can this work? Each random mask produces a smaller subnetwork. Across training, many such subnetworks share and update the same weights. At test time, we disable dropout and use the full network. With inverted dropout, no additional scaling is needed then.

This gives us an interpretation similar to an **ensemble** of subnetworks. The full network approximates their combined predictions; it is not an exact arithmetic average of separately trained models. The [original dropout paper](https://jmlr.org/papers/v15/srivastava14a.html) develops this idea.

How do you think we choose the dropout probability? Exactly: if you read the previous article, you know the answer—using a validation set or cross-validation. There are also more advanced variations, such as *variational dropout*, which I might cover in a dedicated article.

## Activation functions

![The sigmoid activation function](/assets/images/ml05-sigmoid.png)

I wrote about activation functions in previous articles, but we have not explored their use in hidden layers in much depth.

By now, you should understand that we train these networks using gradient descent. Step and sign functions do not work well with ordinary backpropagation because their derivatives are zero almost everywhere and undefined at their jumps. They give gradient descent no useful signal for adjusting the weights.

Sigmoid and tanh are differentiable, but their derivatives can become very small. The sigmoid derivative is at most $1/4$, while the tanh derivative is at most $1$ and approaches zero in its saturated regions.

Why is this a problem? Look at the backpropagation formula for a network with one hidden layer:

$$
\begin{aligned}
\frac{\partial E}{\partial w_{ji}^{(1)}}
&=-2\sum_{n=1}^{N}\left[t_n-g_1(\mathbf{x}_n,\mathbf{w})\right]\\
&\qquad\cdot f_{\mathrm{out}}'(z_{\mathrm{out},n})\,
w_{1j}^{(2)}\,f_{\mathrm{hid}}'(z_{j,n})\,x_{i,n}.
\end{aligned}
$$

Here, $z_{j,n}$ and $z_{\mathrm{out},n}$ are the weighted sums entering the hidden and output units, before their activation functions $f_{\mathrm{hid}}$ and $f_{\mathrm{out}}$.

More layers mean more activation derivatives—and weights—in the chain. Repeatedly multiplying by small factors can make the result extremely small. For example, multiplying ten factors of $0.1$ gives $10^{-10}$.

The layers closest to the input are reached only after the gradient has passed backwards through all the later layers. If it shrinks along the way, those early layers receive tiny gradients and therefore tiny weight updates. The network can still produce outputs, and later layers may still learn, while the early layers learn very slowly. This is the **vanishing gradient problem**.

The weights also affect the size of these products, so activation derivatives alone do not determine the outcome. The issue is particularly important in recurrent neural networks, which we will discuss later.

Let’s introduce another activation function: the **Rectified Linear Unit**, or **ReLU**.

![ReLU returns zero for negative inputs and the input itself for positive inputs](/assets/images/ml05-relu.png)

$$
f(z)=\operatorname{ReLU}(z)=\max(0,z).
$$

Its derivative, away from zero, is:

$$
f'(z)=
\begin{cases}
0,&z<0,\\
1,&z>0.
\end{cases}
$$

Why introduce it? What advantages does it have?

- **It can make training faster.** In a well-known experiment reported in the [AlexNet paper](https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), a network with ReLUs reached a given training error on CIFAR-10 six times faster than an equivalent tanh network. This was an experimental result, not a universal speedup.
- **It can produce sparse activations.** Units receiving negative weighted inputs output zero.
- **It is inexpensive to compute.** Both the function and its derivative are simple.
- **It helps gradient propagation on its positive side.** Its derivative is $1$ there, so it avoids the shrinking factor caused by sigmoid or saturated tanh derivatives along active paths. Other causes of vanishing gradients can still remain.

Clearly, perfection does not exist, so there are disadvantages as well:

- **It is not differentiable at zero.** Implementations choose a derivative value there, commonly zero.
- **Its output is not centered around zero.** ReLU outputs are nonnegative.
- **It is unbounded above.** Large positive inputs produce large outputs.
- **It can produce dying neurons.** If a neuron's weighted input stays negative for all training observations, its output and local derivative stay zero. Its incoming weights then receive no gradient through that unit, and it may remain inactive. Poor initialization or large training updates can lead to this situation. A negative weight, or a negative input for just one observation, does not by itself make a neuron dead.

One alternative designed to reduce the dying-neuron problem is **Leaky ReLU**:

$$
f(z)=
\begin{cases}
z,&z\geq0,\\
\alpha z,&z<0,
\end{cases}
\qquad 0<\alpha<1.
$$

A common example uses $\alpha=0.01$. The negative side now has a small, nonzero slope, allowing gradients to pass through it.

![Leaky ReLU retains a small slope for negative inputs](/assets/images/ml05-leaky-relu.png)

There are many other activation functions in the literature. Feel free to explore them.

## Weight initialization

The behavior and final result of gradient descent also depend on **weight initialization**.

What should we avoid?

- **Weights with an excessively large scale:** repeated multiplication by large weights can amplify signals and gradients. With sigmoid or tanh, large weighted inputs can also push activations into saturation and make their derivatives tiny.
- **All-zero initialization:** if weights and biases all start at zero, neurons in the same hidden layer start identically and remain symmetric during training. This prevents them from learning different features. We usually initialize the weights randomly to break this symmetry; biases can often start at zero.
- **Using a fixed variance without considering layer width:** for example, $w\sim\mathcal{N}(0,0.01)$ may work in some settings, but its scale is not adapted to the number of inputs. Through many layers, signals or gradients may shrink or grow, much as they do when activation derivatives repeatedly multiply them by small or large factors.

Two approaches designed to address these issues are **Xavier initialization** and **He initialization**.

### Xavier initialization

The idea behind Xavier initialization, introduced by **Xavier Glorot and Yoshua Bengio**, is to keep signals and gradients at a reasonable scale as they pass through the network.

Suppose we have an input $\mathbf{x}$ with $I$ components and a linear neuron, whose activation is $h_j(z)=z$. With its bias initialized to zero, the output is:

$$
h_j\left(\sum_{i=1}^{I}w_{ji}x_i\right)
=\sum_{i=1}^{I}w_{ji}x_i.
$$

We start with a linear neuron because its output variance is easier to compute. We want to understand whether the variance grows, shrinks, or stays stable across layers.

Assuming $w_{ji}$ and $x_i$ are independent, the variance of one product is:

$$
\begin{aligned}
\operatorname{Var}(w_{ji}x_i)
&=\mathbb{E}[x_i]^2\operatorname{Var}(w_{ji})\\
&\quad+\mathbb{E}[w_{ji}]^2\operatorname{Var}(x_i)\\
&\quad+\operatorname{Var}(w_{ji})\operatorname{Var}(x_i).
\end{aligned}
$$

If inputs and weights both have mean zero, this simplifies to:

$$
\operatorname{Var}(w_{ji}x_i)
=\operatorname{Var}(w_{ji})\operatorname{Var}(x_i).
$$

Now assume that the input components are independent with a common variance, and that the weights are independently drawn with a common variance and are independent of the inputs. The contributions to the sum are then uncorrelated, giving:

$$
\begin{aligned}
\operatorname{Var}\left(\sum_{i=1}^{I}w_{ji}x_i\right)
&=\sum_{i=1}^{I}\operatorname{Var}(w_{ji}x_i)\\
&=I\operatorname{Var}(w_{ji})\operatorname{Var}(x_i).
\end{aligned}
$$

The output variance is the input variance multiplied by $I\operatorname{Var}(w_{ji})$.

To keep it unchanged in this simplified calculation, we want:

$$
I\operatorname{Var}(w_{ji})=1.
$$

Why? We want to avoid signals repeatedly becoming extremely spread out or collapsing into a very narrow range. Keeping the variance stable is a useful starting point.

If you are not convinced, consider the sigmoid activation:

$$
f(z)=\frac{1}{1+e^{-z}}.
$$

![Sigmoid saturates for large positive or negative inputs, where its derivative approaches zero](/assets/images/ml05-sigmoid-derivative.png)

Here, $z$ is the weighted sum entering the current neuron, **before** its activation function. If the variance of these values becomes extremely large, many neurons can receive very large positive or negative inputs. As the plot shows, sigmoid then saturates:

$$
|z|\gg1\quad\Rightarrow\quad f'(z)\approx0.
$$

During backpropagation, we multiply by these derivatives close to zero, which can produce vanishing gradients.

Returning to our calculation, a Gaussian initialization satisfying the forward-propagation condition is:

$$
w_{ji}\sim\mathcal{N}\left(0,\frac{1}{I}\right).
$$

The second parameter is the variance. Since there are $I$ inputs, multiplying $I$ by $1/I$ gives the factor of $1$ we wanted.

So far, we have considered only forward propagation. Glorot and Bengio also considered what happens to the gradients during backpropagation.

For a connection between layer $l$ and layer $l+1$, we have:

$$
\delta_i^{(l)}
=f'(z_i^{(l)})
\sum_{j=1}^{n_{\mathrm{out}}}
w_{ji}^{(l+1)}\delta_j^{(l+1)}.
$$

Remember:

- $\delta_i^{(l)}=\partial E/\partial z_i^{(l)}$ is the error signal for neuron $i$ in layer $l$.
- $w_{ji}^{(l+1)}$ connects neuron $i$ to neuron $j$ in the next layer.
- $n_{\mathrm{out}}$ is the number of neurons in that next layer.

Using the same kind of independence and shared-variance assumptions, and a linear activation so that $f'=1$, we obtain the approximation:

$$
\operatorname{Var}(\delta^{(l)})
\approx n_{\mathrm{out}}\operatorname{Var}(w)
\operatorname{Var}(\delta^{(l+1)}).
$$

Keeping the gradient variance stable suggests $n_{\mathrm{out}}\operatorname{Var}(w)=1$. The forward condition for that same weight matrix suggests $n_{\mathrm{in}}\operatorname{Var}(w)=1$, where $n_{\mathrm{in}}$ is its number of inputs.

When the two layer widths differ, a single weight variance cannot satisfy both conditions exactly. **Xavier initialization** uses a compromise between them. Its Gaussian form is:

$$
w\sim\mathcal{N}\left(0,\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}\right).
$$

This is the usual Xavier, or Glorot, rule. The [original paper](https://proceedings.mlr.press/v9/glorot10a.html) presents the reasoning and a uniform initialization with this variance.

Clearly, the assumption of linearity matters. **He initialization** accounts for ReLU setting negative inputs to zero, rather than treating the activation as linear. This leads to the Gaussian rule:

$$
w\sim\mathcal{N}\left(0,\frac{2}{n_{\mathrm{in}}}\right).
$$

This rule was developed in [He and colleagues' work on rectifiers](https://arxiv.org/abs/1502.01852).

Before going on, I want to clarify a point: Xavier and He initialization can be applied with other activation functions. Their effectiveness depends on the activation, though, because different functions affect the scale of signals and gradients differently. Xavier is commonly associated with tanh-like activations, while He is designed for ReLU-like activations.

## Batch normalization

Centering and scaling inputs can make training easier. **Whitening** goes further: it gives zero-mean inputs with identity covariance, so their variances are one and their components are uncorrelated. Standardizing features individually does not generally remove correlations.

But what happens inside the network, where each layer receives the outputs of the preceding one?

Consider this simple network:

![A feedforward network with two hidden layers between the input and output](/assets/images/ml05-layer-distributions.png)

Look at Layer 2. Its input is not the original dataset. Its input is the output produced by Layer 1:

$$
\mathbf{a}^{(1)}
=f\left(W^{(1)}\mathbf{x}+\mathbf{b}^{(1)}\right).
$$

Now consider what can happen during training:

1. At one iteration, a feature produced by Layer 1 has mean $2$ and variance $1$ across the observations.
2. Gradient descent updates Layer 1's weights.
3. At the next iteration, that feature might have mean $5$ and variance $4$, even for the same observations.

Layer 2 is therefore learning while the distribution of its inputs keeps changing. This is what the original batch normalization paper called **internal covariate shift**.

Why might that make training harder? Imagine that Layer 2 has learned weights that work well when its inputs are concentrated around $2$. After several updates to Layer 1, they become concentrated around $20$. Layer 2 now operates on substantially different inputs. With some activation functions, this can also move neurons into saturated regions.

This was the [original motivation for batch normalization](https://proceedings.mlr.press/v37/ioffe15.html). It is a useful way to introduce the problem, but not a complete explanation of why the method works: [later research](https://papers.neurips.cc/paper_files/paper/2018/hash/905056c1ac1dad141560467e0a99e1cf-Abstract.html) also highlighted its effect on making optimization smoother.

The point is that normalizing only the original inputs does not keep the values inside every layer normalized. How can we apply normalization within the network during training?

We use **batch normalization**.

![A fully connected layer followed by batch normalization and ReLU](/assets/images/ml05-batch-normalization.png)

In the arrangement shown here, batch normalization standardizes each neuron's weighted input **before the activation function**, using statistics computed across the current mini-batch. It does not decorrelate the features or fully whiten them.

Let’s explain this carefully. The calculation is not particularly complicated, but it is easy to misunderstand which values we are normalizing.

![A fully connected neural network with several hidden layers](/assets/images/ml05-fully-connected-network.png)

Suppose we are considering the second hidden layer, and both it and the preceding hidden layer have **four neurons**. We train with a **mini-batch of 32 observations**.

The input to the whole layer is a matrix $X$ with 32 rows and four columns. Each row is one observation; each column contains the output of one neuron in the preceding layer. For example:

| Observation | Previous neuron 1 | Previous neuron 2 | Previous neuron 3 | Previous neuron 4 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2 | 10 | 3 | 20 |
| 2 | 4 | 12 | 5 | 24 |
| 3 | 6 | 14 | 7 | 28 |
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |
| 32 | 8 | 16 | 9 | 32 |

The numbers are just an illustration.

Each neuron combines the four input features for each observation. Processing the mini-batch through all four neurons gives:

$$
Z=XW+\mathbf{b}.
$$

Here, $X$ has shape $32\times4$, $W$ has shape $4\times4$, and the bias is added to each row. Therefore, $Z$ also has shape $32\times4$. Each column of $Z$ contains the weighted inputs to **one current-layer neuron**, across all 32 observations.

The matching shapes come from choosing four neurons in both layers. If the current layer had six neurons, $W$ would have shape $4\times6$ and $Z$ would have shape $32\times6$. A single neuron produces one value per observation; the whole layer produces the matrix.

Without batch normalization, we would apply the activation function directly to $Z$. Here, we first standardize **each column separately**:

1. Take the 32 values in the first column of $Z$.
2. Compute their mean and variance.
3. Use those statistics to standardize that column.
4. Repeat for each of the other columns.

We do not pool all $32\times4=128$ values into one mean and variance.

For column $j$, the calculations are:

$$
\begin{aligned}
\mu_j&=\frac{1}{32}\sum_{n=1}^{32}Z_{n,j},\\
v_j&=\frac{1}{32}\sum_{n=1}^{32}(Z_{n,j}-\mu_j)^2,\\
\widehat{Z}_{n,j}&=\frac{Z_{n,j}-\mu_j}{\sqrt{v_j+\varepsilon}}.
\end{aligned}
$$

The small positive constant $\varepsilon$ keeps the division numerically stable, including when the variance is very close to zero.

There is one more part. Batch normalization learns a scale $\gamma_j$ and a shift $\beta_j$ for each column:

$$
Y_{n,j}=\gamma_j\widehat{Z}_{n,j}+\beta_j.
$$

These are **learned parameters**, updated during training along with the weights. They let the network choose a suitable scale and mean instead of forcing every normalized feature to remain at zero mean and unit variance.

We then apply the activation function:

$$
A_{n,j}=f(Y_{n,j}).
$$

The output matrix $A$ has the same shape as $Z$. With a batch of one observation and four neurons in the layer, it has one row containing four values.

So the order in this example is: compute the weighted sums, normalize each column across the mini-batch, apply the learned scale and shift, and finally apply the activation function.

At test time, batch normalization normally uses running estimates of the means and variances collected during training, rather than recomputing them from the test observations.

One final point: batch normalization can make training less sensitive to weight initialization by helping control the scale of these intermediate values. A sensible initialization still matters, though; normalization does not guarantee that every signal and gradient throughout the network will behave well.

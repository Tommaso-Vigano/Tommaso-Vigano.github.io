---
layout: post
title: "Machine Learning from Scratch 04: Training and Overfitting in Neural Networks"
date: 2026-10-19
description: "How to recognize overfitting, separate training from model selection, and use early stopping and weight regularization to improve generalization."
categories: [machine-learning, neural-networks, fundamentals]
image: /assets/images/ml04-underfitting-overfitting.png
---

{% assign approximation_post = site.posts | where_exp: "post", "post.title contains 'Machine Learning from Scratch 02'" | first %}

Remember the universal approximation theorem mentioned in {% if approximation_post %}[one of the previous articles]({{ approximation_post.url | relative_url }}){% else %}one of the previous articles{% endif %}? Being able to approximate a function is useful only if the model also generalizes to new observations. This brings us to the problem of **overfitting**.

## Overfitting

![Models that underfit the observations, capture their underlying pattern, or overfit their fluctuations](/assets/images/ml04-underfitting-overfitting.png)

Models that are too simple can **underfit** the data. Models that are too flexible can fit the training observations—including their noise—without generalizing well to new data. This is **overfitting**.

The picture gives us the idea, but we are quantitative people, I hope. More generally, it is not always easy to plot and visualize our data: we often work with points in very high-dimensional spaces. So how do we measure generalization? How do we spot possible overfitting?

We compare the **training loss** with the loss on independent, held-out observations. I’ll try to use *loss* instead of *error*, but I cannot promise anything.

A very low training loss is not a problem by itself. The warning sign is a large gap: the model performs well on the training data but poorly on comparable data it has not learned from. During model development, we monitor this using a **validation set**. We reserve the **test set** for the final assessment.

![Training loss continues to decrease while validation loss eventually starts increasing](/assets/images/ml04-training-validation-loss.png)

To obtain held-out data, we usually split the available dataset. For small datasets, resampling methods can help us make better use of the observations. **Bootstrapping**, for example, uses [sampling with replacement](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html). Repeating observations does not create additional independent test data, though: training and assessment samples must still be kept separate.

For classification, we often use **stratified sampling**, which approximately preserves the class proportions in each split. In the standard i.i.d. setting, we assume that training and future observations come from the same distribution. Stratification helps preserve class balance, but does not guarantee that every aspect of the distributions matches.

While reviewing this part of the post, I noticed that I had moved rather quickly through the notions of training, test, and validation sets. For a beginner, the terminology can be a mess, so let me be clearer.

![The available dataset is separated into development data and a test set; development data is divided into training and validation sets](/assets/images/ml04-dataset-splits.png)

This should help clarify the roles:

- **Available dataset:** all the observations we have before splitting. The diagram calls this the “Training Dataset.”
- **Training set:** the observations used to learn the model’s parameters, such as the network weights.
- **Test set:** the observations used for the final model assessment.
- **Validation set:** the observations used for model selection, such as choosing the network size, activation function, or regularization strength. They do not enter the weight updates during that training run.

The diagram uses **training data** for the development pool, which contains both the training and validation sets. Terminology varies across sources, but the roles matter: fitting parameters, selecting the model, and assessing the final result are different tasks.

How can we check whether a model works well on observations it has not learned from?

The simplest approach is to split our development data into two parts: one for training and one for validation. We keep the test set separate for the final assessment.

There are three common ways to organize training and validation:

- **Holdout:** make one split. Train the model on the training set and measure its loss on the validation set. This is simple, but a different split might give us a different result.
- **Leave-one-out cross-validation (LOOCV):** leave out one observation, train on all the others, and validate on the observation left out. Repeat this for every observation. With $N$ observations, this means training $N$ models, which can be expensive.
- **$K$-fold cross-validation:** divide the data into $K$ groups, called *folds*. Take turns using one group for validation and all the others for training.

![Five-fold cross-validation rotates the validation fold while using the remaining folds for training](/assets/images/ml04-k-fold.png)

Let’s make the last method more concrete. Suppose we have **100 observations** and choose **five-fold cross-validation**. We divide them into five groups of 20 observations.

First, we train on 80 observations and validate on the remaining 20. Then we train a fresh model, holding out a different group of 20. We repeat this five times, so every group gets a turn as the validation set.

Each run gives us a validation loss. To summarize the five results, we simply take their average:

$$
\text{Average validation loss}
=\frac{e_1+e_2+e_3+e_4+e_5}{5},
$$

where $e_1,\ldots,e_5$ are the losses from the five runs. This makes our comparison less dependent on one particular split.

For example, suppose we want to choose between a network with 10 hidden neurons and one with 20. We perform the same cross-validation procedure for each choice and compare their average validation losses.

Once we have selected the configuration, we train a new model using all the development data. Only then do we evaluate it on the untouched test set.

To be more quantitative: let $\mathcal{V}_k$ contain the indices of the validation observations in fold $k$, and let $\mathbf{w}^{(k)}$ be the weights learned without that fold. If $L(t_n,g(\mathbf{x}_n,\mathbf{w}^{(k)}))$ is the loss for observation $n$, then:

$$
\widehat{e}_k
=\frac{1}{|\mathcal{V}_k|}
\sum_{n\in\mathcal{V}_k}
L\left(t_n,g(\mathbf{x}_n,\mathbf{w}^{(k)})\right).
$$

The average cross-validation loss is:

$$
\widehat{E}_{\mathrm{CV}}
=\frac{1}{K}\sum_{k=1}^{K}\widehat{e}_k.
$$


## How to address overfitting in neural networks

We have discussed how to detect overfitting, but not how to reduce it.

One technique is **early stopping**, a common choice when training neural networks.

What is early stopping?

Suppose that, after each epoch, we plot the training and validation losses. At some point, the training loss may keep decreasing while the validation loss starts increasing. Continuing to improve the training fit is then no longer improving performance on the held-out observations.

![Training loss decreases while validation loss reaches a minimum and then increases](/assets/images/ml04-early-stopping.png)

Early stopping uses the validation loss to decide when to stop. In practice, we usually allow a certain number of epochs without improvement—called *patience*—rather than stopping after the first increase. We keep the checkpoint with the best validation loss. The test set does not take part in this decision.

Another technique is **weight decay**: limiting overfitting through weight regularization. Here, we will discuss its usual connection to **L2 regularization** under ordinary gradient descent.

Regularization limits the model’s freedom by introducing a preference for certain solutions. Let’s make that more concrete. In the previous post, we chose the weights by maximizing the data likelihood:

$$
\mathbf{w}_{\mathrm{MLE}}
=\underset{\mathbf{w}}{\operatorname{arg\,max}}\;
p(\mathcal{D}\mid\mathbf{w}).
$$

We can introduce a Bayesian perspective and look for the **maximum a posteriori**, or **MAP**, estimate instead:

$$
\begin{aligned}
\mathbf{w}_{\mathrm{MAP}}
&=\underset{\mathbf{w}}{\operatorname{arg\,max}}\;
p(\mathbf{w}\mid\mathcal{D})\\
&=\underset{\mathbf{w}}{\operatorname{arg\,max}}\;
p(\mathcal{D}\mid\mathbf{w})p(\mathbf{w}).
\end{aligned}
$$

The prior $p(\mathbf{w})$ expresses our assumptions about which weight values are more plausible before observing the data.

Why can this help? Two weight configurations may fit the training data similarly, while one relies on much larger weights. A prior favoring smaller weights makes us prefer the other solution unless the improvement in fit justifies those larger values. This can reduce fitting to noise, although it does not guarantee good generalization.

Work from the early 1990s, such as [Krogh and Hertz’s study of weight decay](https://proceedings.neurips.cc/paper/1991/hash/8eefcfdf5990e441f0fb6f3fad709e21-Abstract.html), explored how penalizing large weights can improve generalization. To encourage weights towards zero, we can assume independent Gaussian priors:

$$
w_q\sim\mathcal{N}(0,\sigma_w^2),
\qquad q=1,\ldots,Q,
$$

where $Q$ is the number of weights being regularized.

Let’s suppose we are doing **regression**, with the same independent Gaussian noise of fixed variance $\sigma^2$ used in the previous post. Why specify regression? Because that likelihood gives us the squared-error fitting term. The Gaussian prior can also be used for classification, but the fitting term would then come from the classification likelihood.

The likelihood and prior are:

$$
\begin{aligned}
p(\mathcal{D}\mid\mathbf{w})
&=(2\pi\sigma^2)^{-N/2}\\
&\qquad\cdot\exp\left[-\frac{1}{2\sigma^2}\sum_{n=1}^{N}
[t_n-g(\mathbf{x}_n,\mathbf{w})]^2\right],\\[6pt]
p(\mathbf{w})
&=\prod_{q=1}^{Q}\frac{1}{\sqrt{2\pi\sigma_w^2}}
\exp\left[-\frac{w_q^2}{2\sigma_w^2}\right].
\end{aligned}
$$

Taking the negative logarithm of their product and dropping terms that are constant with respect to the weights gives:

$$
\begin{aligned}
\mathbf{w}_{\mathrm{MAP}}
=\underset{\mathbf{w}}{\operatorname{arg\,min}}\;\Bigg[
&\frac{1}{2\sigma^2}\sum_{n=1}^{N}
[t_n-g(\mathbf{x}_n,\mathbf{w})]^2\\
&+\frac{1}{2\sigma_w^2}\sum_{q=1}^{Q}w_q^2
\Bigg].
\end{aligned}
$$

Multiplying the objective by the positive constant $2\sigma^2$ does not change its minimizer. Defining $\gamma=\sigma^2/\sigma_w^2$, we obtain:

$$
\begin{aligned}
\mathbf{w}_{\mathrm{MAP}}
=\underset{\mathbf{w}}{\operatorname{arg\,min}}\;\Bigg[
&\underbrace{\sum_{n=1}^{N}[t_n-g(\mathbf{x}_n,\mathbf{w})]^2}_{\text{fitting}}\\
&+\underbrace{\gamma\sum_{q=1}^{Q}w_q^2}_{\text{regularization}}
\Bigg].
\end{aligned}
$$

The first term measures the fit; the second is the regularization term. We have a new contribution to the loss that penalizes large weight magnitudes.

A smaller prior variance $\sigma_w^2$ means stronger shrinkage towards zero. In practice, we usually tune the combined regularization strength $\gamma$. 

![An illustrative validation-loss curve for different regularization strengths, with its lowest displayed value at gamma equal to five](/assets/images/ml04-regularization-selection.png)

How do we choose $\gamma$? We try different values and check which one gives the best predictions on the validation set.

First, split the development data into training and validation sets. For each value of $\gamma$, train a fresh model by minimizing:

$$
E_{\mathrm{TRAIN}}(\mathbf{w})
=\sum_{n=1}^{N_{\mathrm{TRAIN}}}
[t_n-g(\mathbf{x}_n,\mathbf{w})]^2
+\gamma\sum_{q=1}^{Q}w_q^2.
$$

This balances two goals: fitting the training observations and keeping the weights small. A larger $\gamma$ gives more importance to the weight penalty.

After training each model, measure its prediction loss on the validation set:

$$
E_{\mathrm{VAL}}
=\sum_{n=1}^{N_{\mathrm{VAL}}}
[t_n-g(\mathbf{x}_n,\mathbf{w})]^2.
$$

Here, $\mathbf{w}$ contains the weights learned by that model. Notice that **we do not add the weight penalty to the validation loss**: we want to see how well the model predicts observations it has not learned from.

Choose the value of $\gamma$ that gives the lowest validation loss. Then train a new model on the combined training and validation data using that value.

Finally, evaluate the resulting model on the untouched test set.


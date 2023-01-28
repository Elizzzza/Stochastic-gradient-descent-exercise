# Stochastic-gradient-descent-exercise

#### BIOST 561 Computational skills for biostatistics
#### Instructor: Eardi Lila

# Stochastic gradient descent exercise

In class, we have introduced one of the most popular optimization techniques: gradient descent. However,
for problems with very large n, gradient descent can be inefficient and stochastic gradient descent (a.k.a.
batched gradient descent) is instead typically used. The main difference is that stochastic gradient descent,
at each iteration, uses only random subsets of the n training sample to compute the gradient and update the
parameter of interest (&beta; in our problem).

In this problem, you will modify the **gradient descent algorithm for linear regression** introduced in
class (you can also work on the logistic regression problem – slightly more challenging) to perform stochastic
gradient descent optimization.

**Pseudocode for gradient descent** 
Let `x` be a n &times; (p + 1) data-matrix and let `y` be the associated vector of outcomes of length n. 
We assume a linear relationship between input and output. 
Let `beta = beta_init` be the parameter to be optimized and let `niter` be the desired number of iterations.

`for it = 1, 2, 3, ..., niter:`

• Compute the loss function (using the entire training set)

• Compute the gradient of the loss function w.r.t. `beta` (using the entire training set)

• Update parameters:
– `beta = beta – learning_rate*gradient`

**Pseudocode for stochastic gradient descent**  
Let `x` be a n &times; (p + 1)
data-matrix and let `y` be the associated vector of outcomes of length n. We assume a linear relationship
between input and output. Let `beta = beta_init` be the parameter to be optimized and let `niter` be the
desired number of iterations.

`for it = 1, 2, 3, ..., niter:`

• Split the n samples in the training set (same split for both `x` and `y`) into ***B*** groups that we call
mini-batches (Similarly to what you do with k-fold cross-validation).

• for `mini_batch` = 1, 2, . . . , B:

– Compute the loss function (using the entire training set)

– Compute gradient of the loss function w.r.t. beta (using onlythe observations in the current
`mini_batch`)

– Update parameters:
∗ `beta = beta – learning_rate*gradient`

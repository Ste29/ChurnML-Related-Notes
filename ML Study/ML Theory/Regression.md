# Regression

## Table of Contents
* [Ridge Regression](#ridge-regression)
* [Annex](#annex)

## Least Squares Linear Regression
- https://towardsdatascience.com/least-squares-linear-regression-in-python-54b87fc49e77?gi=c33f8d372867
- Da approfondire la parte di matematica!


This technique minimizes the sum of the squares of the residuals between the observed targets.

Scikit learn minimize something similar to this, using the singular value decomposition of X. Also it uses the L2 norm
$$\underset{w}{min}||Xw-y||^2_2$$
This is similar to mean squared error $$MSE= \frac{1}{N}\sum^N_{i=1}(y-y')^2$$
In particular scikit learn calculates the magnitude of x as you can see in the first formula:
$$ ||x|| := \sqrt {x_1^2+x_2^2+...+x_n^2}$$

So formula 1 is how you compute the error, in order to minimize it the pseudoinverse of the input matrix X must be used.
$$w = X^+y$$
$$X^+=VD^+U^T$$
The above matrices are obtained from SVD of X:
$X=U\Sigma V^T$

``` python
from sklearn.datasets import make_regression  
from matplotlib import pyplot as plt  
import numpy as np  
from sklearn.linear_model import LinearRegression

# Synthetic data
X, y, coefficients = make_regression(  
    n_samples=50,  
    n_features=1,  
    n_informative=1,  
    n_targets=1,  
    noise=5,  
    coef=True,  
    random_state=1  
)

# Number of columns and rank
n = X.shape[1]  
r = np.linalg.matrix_rank(X)

# SVD
U, sigma, VT = np.linalg.svd(X, full_matrices=False)

# D^+ calculation
D_plus = np.diag(np.hstack([1/sigma[:r], np.zeros(n-r)])) 
# V is the transpose of its transpose
V = VT.T
# Moore-Penrose pseudoinverse of X
X_plus = V.dot(D_plus).dot(U.T)

# Vector of coefficients
w = X_plus.dot(y)

# Residuals using the first equation
error = np.linalg.norm(X.dot(w) - y, ord=2) ** 2


# ## Now test it against numpy ## #

np.linalg.lstsq(X, y)

# plot
plt.scatter(X, y)  
plt.plot(X, w*X, c='red')

# ## now the same vs scikit-learn ## #

lr = LinearRegression()
lr.fit(X, y)
w = lr.coef_[0]
```

## Ridge Regression
- https://towardsdatascience.com/ridge-regression-python-example-f015345d936b?gi=f6e4943b2dc0

Overfitting is the process by which a model performs well for training samples but fails to generalize. In order to tackle this problem many techniques were implemented, one of them is regularization.

Ridge regression is related to linear regression and it can be used to determine the best fitting line.

**Bias**:
Is the extent to which the model fails to come up with a plot that approximates the samples (if it does not capture the underlying trend in the data).

**Variance**:
Variance does not refer the spread of data relative to the mean. It characterizes the difference in fits between datasets, it measures how the accuracy of the model changes when presented with a different dataset.
Imagine a linear regression vs a regression with lot of curves, the first one has low variance because its MSE would be similar for every dataset, the first one instead could perform very well with certain dataset and very bad with most of the others.

**Ridge regression is similar to linear regression, except a small amount of bias is introduced in order to get a lower variance.**
It performs worse on the specific case, but it is able to generalize better. This is a form of regularization.

A regularization term is added to the loss function. The loss is the **linear least squares function** and the regularization is given by the **L2 norm**.
$$min\ \sum_i^n(y_{pred}-y)^2 + \alpha w^2_0 + \alpha w^2_1 + \alpha w^2_3 + ...$$
Since we want to minimize the loss function and W is included in residual sum of squares, the model is forced to find a balance between minimizing the residual sum of squares and minimizing the coefficients.
$$\sum^n_i(y_{pred}-y)^2 = \sum_i^n(wx-y)^2$$
ypred is the same as wx.
Therefore, in high degree polynomial the coefficients of the higher order variables will tend towards 0 if the underlying data can be approximated just as well with a low degree polynomial.![[ridge.png]]
With a very big $\alpha$ the algorithm will set every weight to 0, giving a line with slope = 0.

$L(w,\alpha) = ||Xw-y||^2 + \alpha||w||^2$ 
$=(Xw - y)^T(Xw-y) +\alpha w^Tw$
$=(w^TX^T-y^T)(Xw-y+\alpha w^Tw)$
$=w^TX^TXw-2w^TX^Ty+y^Ty+\alpha w^Tw$

Assuming $w^Tw$ --> $w^2$ and taking partial derivative with respect to **w** the result is:
$w=(X^TX+\alpha I)X^Ty$

### Code

``` python
from sklearn.datasets import make_regression  
from matplotlib import pyplot as plt  
import numpy as np  
from sklearn.linear_model import Ridge

# data generated suited for regression
X, y, coefficients = make_regression(  
    n_samples=50,  
    n_features=1,  
    n_informative=1,  
    n_targets=1,  
    noise=5,  
    coef=True,  
    random_state=1  
)

# Alpha hyperparameter, larger value = stronger regularization, higher alpha higher bias. Alpha = 1 is linear regression
alpha = 1

# Identity matrix with size of the matrix X^T * X
n, m = X.shape  
I = np.identity(m)

# The last equation presented in theory, also note that w is very close to coefficients
w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + alpha * I), X.T), y)

# Plot of the regression vs scatter of the generated data
plt.scatter(X, y)  
plt.plot(X, w*X, c='red')

# Now the same is done with Ridge Regression from scikit-learn, note that the new w is equal to the previous one
rr = Ridge(alpha=1)
rr.fit(X, y)
w = rr.coef_

# Plot again
plt.scatter(X, y)  
plt.plot(X, w*X, c='red')

# Now visualize the effect of the regularization parameter, the higher the alpha the more horizontal is the line regression
rr = Ridge(alpha=10)
rr.fit(X, y)
w = rr.coef_[0]

plt.scatter(X, y)  
plt.plot(X, w*X, c='red')
``` 

## Annex
Cool story about diamond history and diamond dataset
- https://towardsdatascience.com/know-your-data-pricing-diamonds-using-scatterplots-and-predictive-models-6cce92d794c1

Often the distribution of any monetary variable will be highly skewed and vary over orders of magnitude. This can result from path-dependence (e.g., the rich get richer) and/or the multiplicitive processes (e.g., year on year inflation) that produce the ultimate price/dollar amount. Hence, it’s a good idea to look into compressing any such variable by putting it on a log scale.
Indeed, we can see that the prices for diamonds are heavily skewed, but when put on a log10 scale seem much better behaved (i.e., closer to the bell curve of a normal distribution). In fact, we can see that the data show some evidence of bimodality on the log10 scale, consistent with our two-class, “rich-buyer, poor-buyer” speculation about the nature of customers for diamonds.
Also, since price scales with size and size is the volume which is a function of x\*y\*z you can see a cubic relationship.
# Saddle point approximation

UNDER CONSTRUCTION

This module includes code for saddle point approximation

Saddle point approximations can be used to approximate distributions when you know their cumulant generating function.
We follow the theory as outlined in: Ronald W Butler (2007): "Saddlepoint Approximations with Applications"
As he puts it:

> Among the various tools that have been developed for use in statistics and probability
> over the years, perhaps the least understood and most remarkable tool is the saddlepoint
> approximation. It is remarkable because it usually provides probability approximations
> whose accuracy is much greater than the current supporting theory would suggest.

For a random variable $X$, cumulant generating function $K(t)$ is given by:
$$K(t) = \log\text{E} \exp(t X).$$
Roughly, it contains all information about the random variable's moments, particularly $K'(0)=E X$.
For common distributions (normal, exponential, gamma etc.), the cumulant generating function is known.
Some are implemented as part of this software, others you can find on, e.g., Wikipedia.

Why is this useful? Now, it's not always straightforward to find the distribution of, e.g., the sum of (independent) random variables, as you can't simply sum
the distribution functions. You can, however, sum their cumulant generating functions!
The saddle point approximation can, subsequently be used to approximate the pdf and cdf given the cumulant generating function.

Altogether, applications include:

- Approximating the distribution of the sum of random variables
- Approximating the distribution of the mean of random variables
- Bootstrapping the transform domain: approximating the (bootstrap) distribution of the mean (or some other statistic) on some sample without actually bootstrapping.

To be implemented are:

- Approximating conditional distributions
- Approximating the distribution of the quotient of random variables

## Basic example

To understand what is going on, it is helpful to read the [Wikipedia page](https://en.wikipedia.org/wiki/Saddlepoint_approximation_method) on the saddle point approximation.
The basic steps to use this package are, define the cumulant generating function in terms of the class in this packages, feed it to the saddle point approximation class, evaluate the pdf or cdf.
It's also useful to understand what the saddle point equation is: it's an equation you have to solve to solve for $t$ given $x$.

```python
import numpy as np
import scipy.stats as sps
import spapprox as spa

cgf_normal = spa.norm(loc=0, scale=1)
spa_normal = spa.SaddlePointApprox(cgf_normal)

t = np.linspace(-3, 3)
x = cgf_normal.dK(t)
spa_normal.pdf(t=t)
sps.norm.pdf(x)
```

## Installation

You can install this library directly from GitHub:

```bash
pip install git+https://github.com/mvds314/spapprox.git
```

Optional dependencies are `numdifftools` for numerical differentiation and `fastnorm` for faster bivariate normal cdf evaluation.

````bash

## Development

For development purposes, clone the repo:

```bash
git clone https://github.com/mvds314/spapprox.git
````

Then navigate to the folder containing `setup.py` and run

```bash
pip install -e .
```

to install the package in edit mode.

Run unittests with `pytest`.

Install the optional dependencies to test all functionality.

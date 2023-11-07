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
Some are implemented as part of this software, others you can find on, e.g., Wikepdia.

Why is this useful? Now, it's not always straigtforward to find the distribution of, e.g., the sum of (independent) random variables, as you can can't simply sum
the distribution functions. You can, however, sum their cumulant generating functions!
The saddle point approximation can, subsequently be used to approximate the pdf and cdf given the cumulant generating function.

Altogether, applications include:
* Approximating the distribution of the sum of random variables
* Approximating the distribution of the mean of random variables
* Bootstrapping the transform domain: approximating the (bootstrap) distribution of the mean (or some other statistic) on some sample without actually bootstrapping.

To be implemented are:
* Approximating conditional distributions
* Approximating the distirbution of the quotient of random variables

## Installation

```python
import numpy as np
#TODO: put example here
```

## Installation

You can install this library directly from github:

```bash
pip install git+https://github.com/mvds314/spapprox.git
```

## Development

For development purposes, clone the repo:

```bash
git clone https://github.com/mvds314/spapprox.git
```

Then navigate to the folder containing `setup.py` and run

```bash
pip install -e .
```

to install the package in edit mode.

Run unittests with `pytest`.

Install the optional dependencies to test all functionality.

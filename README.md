# statistics

The goal of this package is to explore how a statistics API may simplify statistical modeling with Python. Several aspects of statistics are manually coded by several users multiple times or treated as black-boxes over several statistics packages. A primary focus in this codebase is how can we develop a simple API that facilitates numerical analysis, without *hiding* the underlying functions and assumptions within.

A key focus of to this end is how statistical distributions can be treated as numerical objects. Take a linear function with a white noise property. `stats.distributions` simplifies the functional form of this by making numerical coding mathematical.

```python
import numpy as np
from stats import Normal

# Create a Normal distribution object
N = Normal(0, 1, seed=42)

# Create empty array
X = np.zeros(10)

# Add noise
X + N
>>> array([ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337,
       -0.23413696,  1.57921282,  0.76743473, -0.46947439,  0.54256004])
```

Here the sampling is performed automatically without a loss of understanding.
# Neigbhorhood Component Feature Selection

This is a Python implementation of Neighborhood Component Feature Selection,
originally introduced in [Yang et al. 2012](http://www.jcomputers.us/vol7/jcp0701-19.pdf).
NCFS is an embedded feature selection method that learns feature weights by
maximizing prediction accuracy in a leave-one-out KNN classifier.

## Installation

The package is not currently available on `pip` or `conda`, so cloning/forking
the repository is the best way to install the package. The package relies on
`numpy` and `scipy`. 

## Example

```python
from NCFS import NCFS

X, y = NCFS.toy_dataset()
feature_select = NCFS.NCFS()
feature_select.fit(X, y)
print(sum(feature_select.coef_ > 1))
```

## Tests

Unit tests are currently under development, but results comparing to the
original paper are listed below. To generate plots yourself, run
`python tests/generate_results.py`

## Comparison with Original Results

### Distance metric

The original paper uses the Manhattan distance `(1)` when calculating distances
between samples/features. While this implementation defaults to using this
distance, weights comparable with published results were only found using the
euclidean distance. However, while exact weights differed between distance
metrics, the selected features did not. Unfortunately, the original paper
did not link to the code used, and I've been unable to find a public
implementation of the aglorithm.

### Numerical stability
![Formulas](/images/distance.png)

NCFS uses the kernal function `(2)` when calculating probabilities. However, with
a large number of features, the value `z` -- canonically a distance value from
`(1)` -- can easily approach a large enough value such that the negative
exponent rounds to zero. This leads to division by zero issues, and fitting
fails. To get around this, small pseudocounts are added to distances when a
division by zero would otherwise occur. To keep distances small, features should
be scaled between 0 and 1 (enforced by NCFS), and again, use of the euclidean
distance is recommended.

### Selected Features
Figure 1, Yang et al. 2012
![Figure1](/images/Figure1.png)

Figure 1 Comparison
![Comparison1](/images/figure1_comp.png)

Figure 2, Yang et al. 2012
![Figure2](/images/Figure2.png)

Figure 2 Comparison
![Comparison2](/images/figure2_comp.png)

Figure 3, Yang et al. 2012
![Figure3](/images/Figure3.png)

Figure 3 Comparison
![Comparison3](/images/figure3_comp.png)

# Neigbhorhood Component Feature Selection

This is a Python implementation of Neighborhood Component Feature Selection,
originally introduced in [Yang et al. 2012](http://www.jcomputers.us/vol7/jcp0701-19.pdf).
NCFS is an embedded feature selection method that learns feature weights by
maximizing prediction accuracy in a leave-one-out KNN classifier.

## Installation

The package can be with pip using the following command:

`pip install ncfs`

## Example

```python
from ncfs import NCFS, toy_dataset

X, y = toy_dataset()
feature_select = NCFS()
feature_select.fit(X, y)
print(sum(feature_select.coef_ > 1))
```

## Tests

Unit tests for distances can be run with the following command:

`tests/test_distances.py`

To generate plots comparing results from the original publicaation, run:

`python tests/generate_results.py`

## Integration with Scikit-Learn

The main `NCFS` class extends base `Estimator` and `Transformer` classes from
`scikit-learn`, and thus can take advantage of the supporting functions in the
`scikit-learn` library.

## Comparison with Original Paper

### Distance metric

The original paper uses the Manhattan distance when calculating distances
between samples/features. While this implementation defaults to using this
metric, the exact weights differed between reported values. However, the
*selected* features did not. Unfortunately, the original paper did not link to
the code used, and I've been unable to find a public implementation of the
aglorithm.

### Numerical stability

NCFS uses the original kernel function when calculating probabilities; however, with
a large number of features, distance values can easily approach a large enough
value such that the negative exponent rounds to zero. This leads to division by
zero issues, and fitting fails. To get around this, small pseudocounts are added
to distances when a division by zero would otherwise occur. To keep distances
small, features should be scaled between 0 and 1 (enforced by NCFS).


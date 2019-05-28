#include <string>
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"


/*!
Validate vectors are 1 dimensional.
*/
inline xt::xarray<double> _validate_vector(xt::xarray<double> x) {
    auto out = xt::squeeze(x);
    if (out.dimension() > 1) {
        throw "Expected 1-dimensional array.";
    }
    return out;
}

/*!
Check vectors are the same shape
*/
inline void _check_shape(xt::xarray<double> x,
                         xt::xarray<double> y) {
    if (x.shape() != y.shape()) {
        throw "`x` and `y` are different shapes";
    }
}

/*!
Calculate the minkowski distance between two vectors of the same size.

@param x: first (n x 1) array
@param y: second (n x 1) array
@param p: scalar value to raise values to.
@param w: optional weight parameter, either an (n x 1) vector for individual
    weights for each feature, or a single scalar value to apply to all weights.

@return minkowski distance between x and y
*/
xt::xarray<double> minkowski(xt::xarray<double> x, xt::xarray<double> y,
                             double p, xt::xarray<double> w);
xt::xarray<double> minkowski(xt::xarray<double> x, xt::xarray<double> y,
                             double p, double w=1);
xt::xarray<double> minkowski(xt::xarray<double> x, xt::xarray<double> y,
                             double p);

/*!
Calculate the squared euclidean distance between two vectors of the same size.

@param x: first (n x 1) array
@param y: second (n x 1) array
@param w: optional weight parameter, either an (n x 1) vector for individual
    weights for each feature, or a single scalar value to apply to all weights.

@return squared euclidean distance between x and y
*/
xt::xarray<double> sqeuclidean(xt::xarray<double> x, xt::xarray<double> y,
                               xt::xarray<double> w);
xt::xarray<double> sqeuclidean(xt::xarray<double> x, xt::xarray<double> y,
                               double w=1);
xt::xarray<double> sqeuclidean(xt::xarray<double> x, xt::xarray<double> y);

/*!
Calculate the pairwise distance matrix between samples in a data matrix.

@param X: an (n x p) data matrix.
@param metric: distance metric to to use. Options are 'minkowski' or 
    'sqeuclidean'.
@param w: an optional (1 x p) vector of feature weights.

@return An (n x n) distance matrix.
*/
xt::xarray<double> pdist(xt::xarray<double> X, std::string metric,
                         xt::xarray<double> w, double p=2);

xt::xarray<double> pdist(xt::xarray<double> X, std::string metric, double w,
                         double p=2);

xt::xarray<double> pdist(xt::xarray<double> X, std::string metric, double p=2);

/*!
Row-center a distance matrix.

Zero center rows in a distance matrix by subtracting the row average from each
value. While this decreases the magnitude in distances, necessary for NCFS, it
does break symmetry. You can no longer assume D[i, j] == D[j, i].

@param dist_mat: an (n x n) distance matrix.

@return An (n x n) row-centered distance matrix. 
*/
xt::xarray<double> center_distances(xt::xarray<double> dist_mat);
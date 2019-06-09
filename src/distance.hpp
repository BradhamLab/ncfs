#include <string>
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include <stdexcept>
#ifndef DISTANCE_H
#define DISTANCE_H

namespace distance {
/*!
Validate vectors are 1 dimensional.
*/
inline xt::xarray<double> _validate_vector(xt::xarray<double> x) {
    auto out = xt::squeeze(x);
    if (out.dimension() > 1) {
        std::string msg = "Expected 1-dimensional array.";
        throw std::runtime_error(msg);
    }
    return out;
}

/*!
Check vectors are the same shape
*/
inline void _check_shape(xt::xarray<double> x,
                         xt::xarray<double> y) {
    if (x.shape() != y.shape()) {
        std::string msg = "`x` and `y` are different shapes";
        throw std::runtime_error(msg);
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
xt::xarray<double> minkowski(const xt::xarray<double>& x,
                             const xt::xarray<double>& y,
                             const double& p,
                             const xt::xarray<double>& w);
xt::xarray<double> minkowski(const xt::xarray<double>& x,
                             const xt::xarray<double>& y,
                             const double& p,
                             const double& w=1);
xt::xarray<double> minkowski(const xt::xarray<double>& x,
                             const xt::xarray<double>& y,
                             const double& p);

/*!
Calculate the squared euclidean distance between two vectors of the same size.

@param x: first (n x 1) array
@param y: second (n x 1) array
@param w: optional weight parameter, either an (n x 1) vector for individual
    weights for each feature, or a single scalar value to apply to all weights.

@return squared euclidean distance between x and y
*/
xt::xarray<double> sqeuclidean(const xt::xarray<double>& x,
                               const xt::xarray<double>& y,
                               const xt::xarray<double>& w);
xt::xarray<double> sqeuclidean(const xt::xarray<double>& x,
                               const xt::xarray<double>& y,
                               const double& w=1);
xt::xarray<double> sqeuclidean(const xt::xarray<double>& x,
                               const xt::xarray<double>& y);

/*!
Calculate the pairwise distance matrix between samples in a data matrix.

@param X: an (n x p) data matrix.
@param metric: distance metric to to use. Options are 'minkowski' or 
    'sqeuclidean'.
@param w: an optional (1 x p) vector of feature weights.

@return An (n x n) distance matrix.
*/
xt::xarray<double> pdist(const xt::xarray<double>& X,
                         const std::string& metric,
                         const xt::xarray<double>& w,
                         const double& p=2);

xt::xarray<double> pdist(const xt::xarray<double>& X,
                         const std::string& metric,
                         const double& w,
                         const double& p=2);

xt::xarray<double> pdist(const xt::xarray<double>& X, 
                         const std::string& metric,
                         const double& p=2);

/*!
Row-center a distance matrix.

Zero center rows in a distance matrix by subtracting the row average from each
value. While this decreases the magnitude in distances, necessary for NCFS, it
does break symmetry. You can no longer assume D[i, j] == D[j, i].

@param dist_mat: an (n x n) distance matrix.

@return An (n x n) row-centered distance matrix. 
*/
xt::xarray<double> center_distances(const xt::xarray<double>& dist_mat);

/*!
Calculate the pairwise distance between each sample in each feature.

@param data: an (n x m) data matrix with n samples and m features.
@param metric: distance metric to use. Either minkowski or sqeucliean.
@param p: power to raise distances to with minkowski distance.
*/
xt::xarray<double> pairwise_feature_distance(const xt::xarray<double>& data_matrix,
                                             const std::string& metric,
                                             const double& p);

} // end namespace
#endif /* DISTANCE_H */
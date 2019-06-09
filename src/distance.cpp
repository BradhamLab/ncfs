#include "distance.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xio.hpp"
#include <iostream>

namespace distance {
xt::xarray<double> minkowski(const xt::xarray<double>& x,
                             const xt::xarray<double>& y,
                             const double& p, 
                             const xt::xarray<double>& w) {
    _check_shape(x, y);
    // x = _validate_vector(x);
    // y = _validate_vector(y);
    // w = _validate_vector(w);
    auto summed = xt::sum(w * xt::pow(xt::abs(x - y), p));
    return xt::pow(summed, 1.0 / p);
}

xt::xarray<double> minkowski(const xt::xarray<double>& x, 
                             const xt::xarray<double>& y,
                             const double& p,
                             const double& w) {
    xt::xarray<double> w_vec = xt::ones_like(x) * w;
    return minkowski(x, y, p, w_vec);
}

xt::xarray<double> minkowski(const xt::xarray<double>& x,
                             const xt::xarray<double>& y,
                             const double& p) {
    return minkowski(x, y, p, 1);
}

xt::xarray<double> sqeuclidean(const xt::xarray<double>& x,
                               const xt::xarray<double>& y,
                               const xt::xarray<double>& w){
    _check_shape(x, y);
    // x = _validate_vector(x);
    // y = _validate_vector(y);
    // w = _validate_vector(w);
    auto x_y = (x - y);
    return xt::sum(w * (x_y * x_y));
}

xt::xarray<double> sqeuclidean(const xt::xarray<double>& x,
                               const xt::xarray<double>& y,
                               const double& w) {
    xt::xarray<double> w_vec = xt::ones_like(x) * w;
    return sqeuclidean(x, y, w_vec);
}
xt::xarray<double> sqeuclidean(const xt::xarray<double>& x,
                               const xt::xarray<double>& y) {
    return sqeuclidean(x, y, 1);
}

xt::xarray<double> pdist(const xt::xarray<double>& X,
                         const std::string& metric,
                         const xt::xarray<double>& w, const double& p) {
    if (X.dimension() != 2) {
        std::string msg = "Expected two-dimensional array for  `X`.";
        throw std::runtime_error(msg);
    }
    if (metric != "minkowski" && metric != "sqeuclidean") {
        std::string msg =  std::string("Unexpected metric: ").append(metric);
        throw std::runtime_error(msg);
    }
    xt::xarray<double> mat = xt::zeros<double>({X.shape(0), X.shape(0)});
    // diagnol in distance matrix should be zero, don't calculate.
    // distance matrix is symmetric, calculate upper triangle only
    for (int i=0; i < X.shape(0); i++) {
        // mat(i, xt::all()) = xt::sum()
        for (int j=i + 1; j < X.shape(0); j++) {
            auto x1 = xt::view(X, i);
            auto x2 = xt::view(X, j);
            double dist;
            if (metric == "minkowski") {
                // must evaulate expression to assign to xarray
                dist = minkowski(x1, x2, p, w)();
            } else {
                // must evaulate expression to assign to xarray
                dist = sqeuclidean(x1, x2, w)();
            }
            mat(i, j) = dist;
            mat(j, i) = dist;
        }
    }
    return mat;
}

xt::xarray<double> pdist(const xt::xarray<double>& X,
                         const std::string& metric, 
                         const double& w, const double& p) {
    xt::xarray<double> w_vec = xt::ones<double>({1, int(X.shape(1))}) * w;
    return pdist(X, metric, w_vec, p);
}

xt::xarray<double> pdist(const xt::xarray<double>& X,
                         const std::string& metric, const double& p) {
    return pdist(X, metric, 1, p);
}

xt::xarray<double> center_distances(const xt::xarray<double>& dist_mat) {
    int n = dist_mat.shape(0);
    auto means = xt::sum(dist_mat, 1) / (n - 1);
    auto center_mat = xt::ones<double>({n, n}) * means;
    return dist_mat - xt::transpose(center_mat);
}

xt::xarray<double> pairwise_feature_distance(const xt::xarray<double>& data_matrix,
                                             const std::string& metric,
                                             const double& p) {
    // a |feature | x |sample| x |sample| distance matrix, where
    // dists[0] is a pairwise distance matrix for samples in
    xt::xarray<double> dists = xt::zeros<double>({int(data_matrix.shape(1)),
                                                  int(data_matrix.shape(0)),
                                                  int(data_matrix.shape(0))});
    for (int i = 0; i < data_matrix.shape(1); i++) {
        auto f_column = xt::reshape_view(xt::view(data_matrix, xt::all(), i),
                                         {int(data_matrix.shape(0)), 1});
        auto f_pdist = pdist(f_column, metric, 1, p);
        for (int k=0; k < f_pdist.shape(0); k++) {
            for (int l=0; l < f_pdist.shape(1); l++) {
                dists(i, k, l) = f_pdist(k, l);
            }
        }
    }
    return dists;
}
} // end namespace
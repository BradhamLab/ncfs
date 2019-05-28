#include "distance.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xio.hpp"
#include <iostream>

xt::xarray<double> minkowski(xt::xarray<double> x, xt::xarray<double> y,
                             double p, xt::xarray<double> w) {
    _check_shape(x, y);
    x = _validate_vector(x);
    y = _validate_vector(y);
    w = _validate_vector(w);
    auto summed = xt::sum(w * xt::pow(xt::abs(x - y), p));
    return xt::pow(summed, 1 / p);
}

xt::xarray<double> minkowski(xt::xarray<double> x, xt::xarray<double> y,
                            double p, double w) {
    xt::xarray<double> w_vec = xt::ones_like(x) * w;
    return minkowski(x, y, p, w_vec);
}

xt::xarray<double> minkowski(xt::xarray<double> x, xt::xarray<double> y,
                             double p) {
    return minkowski(x, y, p, 1);
}

xt::xarray<double> sqeuclidean(xt::xarray<double> x, xt::xarray<double> y,
                               xt::xarray<double> w){
    _check_shape(x, y);
    x = _validate_vector(x);
    y = _validate_vector(y);
    w = _validate_vector(w);
    auto x_y = (x - y);
    return xt::sum(w * (x_y * x_y));
}

xt::xarray<double> sqeuclidean(xt::xarray<double> x, xt::xarray<double> y,
                               double w) {
    xt::xarray<double> w_vec = xt::ones_like(x) * w;
    return sqeuclidean(x, y, w_vec);
}
xt::xarray<double> sqeuclidean(xt::xarray<double> x, xt::xarray<double> y) {
    return sqeuclidean(x, y, 1);
}

xt::xarray<double> pdist(xt::xarray<double> X, std::string metric,
                         xt::xarray<double> w, double p) {
    if (X.dimension() != 2) {
        throw "Expected two-dimensional array for  `X`.";
    }
    if (metric != "minkowski" && metric != "sqeuclidean") {
        throw "Unexpected metric: " < metric;
    }
    xt::xarray<double> mat = xt::zeros<double>({X.shape(0), X.shape(0)});
    // diagnol in distance matrix should be zero, don't calculate.
    // distance matrix is symmetric, calculate upper triangle only
    for (int i=0; i < X.shape(0); i++) {
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

xt::xarray<double> pdist(xt::xarray<double> X, std::string metric, double w,
                         double p) {
    xt::xarray<double> w_vec = xt::ones<double>({1, int(X.shape(1))}) * w;
    return pdist(X, metric, w_vec, p);
}

xt::xarray<double> pdist(xt::xarray<double> X, std::string metric, double p) {
    return pdist(X, metric, 1, p);
}

xt::xarray<double> center_distances(xt::xarray<double> dist_mat) {
    int n = dist_mat.shape(0);
    auto means = xt::sum(dist_mat, 1) / (n - 1);
    auto center_mat = xt::ones<double>({n, n}) * means;
    return dist_mat - xt::transpose(center_mat);
}
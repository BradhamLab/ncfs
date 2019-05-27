#include "distance.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xview.hpp"

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
    auto wx_y = w * (x - y);
    return xt::linalg::dot(wx_y, wx_y);
}

xt::xarray<double> sqeuclidean(xt::xarray<double> x, xt::xarray<double> y,
                               double w=1) {
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
    for (int i=1; i < X.shape(0); i++) {
        for (int j=i; j < X.shape(0); j++) {
            auto x1 = xt::view(X, i);
            auto x2 = xt::view(X, j);
            double dist;
            if (metric == "minkowski") {
                dist = minkowski(x1, x2, p, w)();
            } else {
                dist = sqeuclidean(x1, x2, w)();
            }
            mat(i, j) = dist;
            mat(j, i) = dist;
        }
    }
    return mat;
}

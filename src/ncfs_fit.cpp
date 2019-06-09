#include "distance.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xrandom.hpp"
#include <iostream>
#include <string>
#include "ncfs.hpp"
#include <istream>
#include <fstream>
#include "xtensor/xcsv.hpp"
#include "xtensor/xsort.hpp"

#ifdef XTENSOR_USE_XSIMD
#endif

int main(int argc, char* argv[]) {
    ncfs::NCFS model;
    // xt::xarray<double> X {
    //     {1.0, 2.0},
    //     {2.0, 5.0},
    //     {3.0, 4.0}
    // };
    // xt::xarray<double> y {1, 1, 2};
    // xt::xarray<double> ones {1.0, 1.0};
    // std::string metric = "minkowski";
    // std::cout << "X Input: \n" << X << std::endl;
    // xt::xarray<double> res = distance::pdist(X, metric, ones, 2.0);
    // std::cout << "y Input: \n" << y << std::endl;
    // xt::xarray<double> weights = model.fit(X, y);
    // std::cout << "weights: " << weights << std::endl;

    std::ifstream in_file;
    in_file.open("../data/toy_data_X.csv");
    auto X = xt::load_csv<double>(in_file);
    std::cout << X.shape(0) << ", " << X.shape(1) << std::endl;
    std::ifstream y_file;
    y_file.open("../data/toy_data_Y.csv");
    auto y_ = xt::load_csv<double>(y_file);
    auto y = xt::view(y_, xt::all(), 0);
    // std::cout << y.shape(0)  << ", " << y.shape(1) << std::endl;
    
    xt::xarray<double> weights = model.fit(X, y);
    std::cout << "weights: " << weights << std::endl;
      // print(np.argsort(-1 * f_select.coef_)[:10])
    xt::xarray<double> r_weights = -1.0 * weights;
    xt::xarray<double> sorted = xt::argsort(r_weights);
    std::cout << "sorted: " << sorted << std::endl;
    // for (int i=0; i < 10; i++) {
    //     std::cout << i << ", " << sorted(i) << ", " << weights(sorted(i)) << std::endl;
    // }
    std::cout << "0 weight: " << weights(0) << ", 100 weight: " << weights(100) << std::endl;
    return 0;
}
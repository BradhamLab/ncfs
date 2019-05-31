#include "distance.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xrandom.hpp"
#include <iostream>
#include <string>
#include "ncfs.hpp"

int main(int argc, char* argv[]) {
    xt::xarray<double> arr1 {
        {1.0, 2.0},
        {2.0, 5.0},
        {3.0, 4.0}
    };
    xt::xarray<double> arr2 {1, 1, 2};
    xt::xarray<double> ones {1.0, 1.0};
    std::string metric = "minkowski";
    std::cout << "Input: \n" << arr1 << std::endl;
    xt::xarray<double> res = distance::pdist(arr1, metric, ones, 2.0);
    std::cout << "Output: \n" << res << std::endl;
    xt::xarray<double> fdist = distance::pairwise_feature_distance(arr1, metric, 2.0);
    std::cout << "Centered: \n" << distance::center_distances(res) << std::endl;
    std::cout << "fpdist: \n" << fdist << std::endl;
    ncfs::NCFSOptimizer hey = ncfs::NCFSOptimizer();
    std::cout << "NCFSOptAlpha: " << hey.get_alpha() << std::endl;
    hey.set_alpha(0.5);
    std::cout << "Changed Alpha: " << hey.get_alpha() << std::endl;
    ncfs::KernelMixin oi = ncfs::KernelMixin(1.0, 2.0, xt::ones<double>({2, 1}));
    std::cout << oi.gradients(xt::random::randn<double>({3, 3}),
                              fdist, xt::eye(3)) << std::endl;
    ncfs::NCFS model;
    model.fit(arr1, arr2);
    return 0;
}
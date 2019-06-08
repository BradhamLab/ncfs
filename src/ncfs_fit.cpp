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

int main(int argc, char* argv[]) {
    std::ifstream in_file;
    in_file.open("../data/toy_data_X.csv");
    auto X = xt::load_csv<double>(in_file);
    std::ifstream y_file;
    y_file.open("../data/toy_data_Y.csv");
    auto y = xt::load_csv<double>(y_file);
    ncfs::NCFS model;
    xt::xarray<double> weights = model.fit(X, y);
      // print(np.argsort(-1 * f_select.coef_)[:10])
    auto sorted = xt::argsort(-1 * weights);
    for (int i=0; i < 10; i++) {
        std::cout << i << ", " << sorted(i) << ", " << weights(sorted(i)) << std::endl;
    }
    return 0;
}
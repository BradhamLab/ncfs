#include <string>
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xview.hpp"
// #include "distance.hpp"

class NCFSOptimizer {
    private:
        double alpha;
    public:
        NCFSOptimizer(double a=0.01);
        void set_alpha(double a){alpha = a;};
        double get_alpha(){return alpha;};
        xt::xarray<double> steps(xt::xarray<double>);
        xt::xarray<double> steps(xt::xarray<double>, double);
};

class KernelMixin {
    private:
        double sigma;
        double reg;
        xt::xarray<double> weights;
    public:
        KernelMixin();
        KernelMixin(double sig, double lambda, xt::xarray<double> init_weights);
        xt::xarray<double> transform(xt::xarray<double> dmat);
        xt::xarray<double> gradients(xt::xarray<double>,
                                     xt::xarray<double>,
                                     xt::xarray<double>);
        void set_weights(xt::xarray<double> new_weights){weights=new_weights;};
};


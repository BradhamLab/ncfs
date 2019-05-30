#include <string>
#include <iostream>
#include "xtensor/xio.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xview.hpp"
#include "distance.hpp"

// define namespace
namespace ncfs {
    
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
    protected:
        double sigma;
        double reg;
        xt::xarray<double> weights;
    public:
        KernelMixin(double sig, double lambda, xt::xarray<double> init_weights);
        xt::xarray<double> transform(xt::xarray<double> dmat);
        xt::xarray<double> gradients(xt::xarray<double> p_ref,
                                     xt::xarray<double> f_dist,
                                     xt::xarray<double> class_mat);
        void set_weights(xt::xarray<double> new_weights){weights=new_weights;};
};

class ExponentialKernel : public KernelMixin {
    public:
        using KernelMixin::KernelMixin;
        xt::xarray<double> transform(xt::xarray<double> dmat);
};

class GaussianKernel : public KernelMixin {
    public:
        using KernelMixin::KernelMixin;
        xt::xarray<double> transform(xt::xarray<double> dmat);
};

} // end namespace
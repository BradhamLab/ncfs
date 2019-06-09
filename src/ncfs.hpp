#include <string>
#include <iostream>
#include "xtensor/xio.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xview.hpp"
#include "distance.hpp"
#include "xtensor/xindex_view.hpp"
#include <limits>
#include <cstdlib>
#include "xtensor-blas/xlinalg.hpp"
// define namespace
namespace ncfs {

class NCFSOptimizer {
    private:
        double alpha;
    public:
        NCFSOptimizer(const double& a=0.01);
        void set_alpha(const double& a){alpha = a;};
        double get_alpha(){return alpha;};
        xt::xarray<double> steps(const xt::xarray<double>&);
        xt::xarray<double> steps(const xt::xarray<double>&, const double&);
};

class KernelMixin {
    protected:
        double sigma;
        double reg;
        xt::xarray<double> weights;
    public:
        KernelMixin(const double& sig, const double& lambda,
                    const xt::xarray<double>& init_weights);
        xt::xarray<double> transform(const xt::xarray<double>& dmat);
        xt::xarray<double> gradients(const xt::xarray<double>& p_ref,
                                     const xt::xarray<double>& f_dist,
                                     const xt::xarray<double>& class_mat);
        void set_weights(const xt::xarray<double>& new_weights){weights=new_weights;};
        xt::xarray<double> get_weights(){ return weights;};
};

class ExponentialKernel : public KernelMixin {
    public:
        using KernelMixin::KernelMixin;
        xt::xarray<double> transform(const xt::xarray<double>& dmat);
};

class GaussianKernel : public KernelMixin {
    public:
        using KernelMixin::KernelMixin;
        xt::xarray<double> transform(const xt::xarray<double>& dmat);
};

class NCFS {
    private:
        double alpha_;
        double sigma_;
        double lambda_;
        double eta_;
        std::string metric_;
        std::string kernel_;
        double score_;
        double p_;
        xt::xarray<double> format_class_matrix(const xt::xarray<double>& y);
    public:
        NCFS(double alpha=0.1,
             double sigma=1,
             double lambda=1,
             double eta=1e-10,
             std::string metric="minkowski",
             std::string kernel="gaussian",
             double p=1);
        xt::xarray<double> fit(const xt::xarray<double>& X,
                               const xt::xarray<double>& y);
};

} // end namespace
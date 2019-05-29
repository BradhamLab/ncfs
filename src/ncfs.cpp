#include "ncfs.hpp"

NCFSOptimizer::NCFSOptimizer(double a) {
    this->set_alpha(a);
}

xt::xarray<double> NCFSOptimizer::steps(xt::xarray<double> gradients) {
    return gradients * alpha;
}

xt::xarray<double> NCFSOptimizer::steps(xt::xarray<double> gradients,
                                        double loss){
    auto out = gradients * alpha;
    if (loss > 0) {
        this->set_alpha(alpha * 1.01);
    } else {
        this->set_alpha(alpha * 0.4);
    }
    return out;
}

KernelMixin::KernelMixin(double sig, double lambda,
                         xt::xarray<double> init_weights) {
    sigma = sig;
    reg = lambda;
    set_weights(init_weights);
}

xt::xarray<double> KernelMixin::transform(xt::xarray<double> dmat) {
    return dmat;
}

xt::xarray<double> KernelMixin::gradients(xt::xarray<double> p_ref,
                                         xt::xarray<double> f_dist,
                                         xt::xarray<double> class_mat){
    // initialize weight gradients as zeros
    xt::xarray<double> deltas = xt::zeros_like(weights);
    // calculate the probability of a correct classification
    auto p_correct = xt::sum(p_ref * class_mat, 1);
    // calculate feature gradients
    for (int l=0; l < weights.size(); l++) {
        auto weighted_dist = xt::view(f_dist, l, xt::all(), xt::all()) * p_ref;
        auto all_term = p_correct * xt::sum(weighted_dist, 1);
        auto in_class_term = xt::sum(weighted_dist * class_mat, 1);
        auto sample_terms = all_term - in_class_term;
        deltas(l) = (2 * weights(l)
                  * ((1 / sigma) * xt::sum(sample_terms) - reg))();
    }


    return deltas;
}
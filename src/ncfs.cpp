#include "ncfs.hpp"
#include <chrono>

namespace ncfs {
NCFSOptimizer::NCFSOptimizer(const double& a) {
    this->set_alpha(a);
}

xt::xarray<double> NCFSOptimizer::steps(const xt::xarray<double>& gradients) {
    return gradients * alpha;
}

xt::xarray<double> NCFSOptimizer::steps(const xt::xarray<double>& gradients,
                                        const double& loss){
    auto out = gradients * alpha;
    if (loss > 0) {
        this->set_alpha(alpha * 1.01);
    } else {
        this->set_alpha(alpha * 0.4);
    }
    return out;
}

KernelMixin::KernelMixin(const double& sig, const double& lambda,
                         const xt::xarray<double>& init_weights) {
    sigma = sig;
    reg = lambda;
    set_weights(init_weights);
}

xt::xarray<double> KernelMixin::transform(const xt::xarray<double>& dmat) {
    return dmat;
}

xt::xarray<double> KernelMixin::gradients(const xt::xarray<double>& p_ref,
                                          const xt::xarray<double>& f_dist,
                                          const xt::xarray<double>& class_mat){
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
        // force evaluation
        deltas(l) = (2.0 * weights(l)
                  * ((1.0 / sigma) * xt::sum(sample_terms) - reg))();
    }


    return deltas;
}

xt::xarray<double> ExponentialKernel::transform(const xt::xarray<double>& dmat) {
    return xt::exp(-1.0 * dmat * (1.0 / sigma));
}

xt::xarray<double> GaussianKernel::transform(const xt::xarray<double>& dmat) {
    auto centered = distance::center_distances(dmat);
    return xt::exp(-1.0 * centered * (1.0 /  sigma));
}
NCFS::NCFS(double alpha, double sigma, double lambda, double eta,
             std::string metric, std::string kernel, double p) {
    alpha_ = alpha;
    sigma_ = sigma;
    lambda_ = lambda;
    eta_ = eta;
    metric_ = metric;
    kernel_ = kernel;
    p_ = p;
}

xt::xarray<double> NCFS::format_class_matrix(const xt::xarray<double>& y) {
    // y = distance::_validate_vector(y);
    xt::xarray<double> class_matrix = xt::zeros<double>({y.size(), y.size()});
    for (int i = 0; i < y.size(); i++) {
        for (int j = i + 1; j < y.size(); j++) {
            if (y(i) == y(j)) {
                class_matrix(i, j) = 1;
                class_matrix(j, i) = 1;
            }
        }
    }
    return class_matrix;
}

xt::xarray<double> NCFS::fit(const xt::xarray<double>& X,
                             const xt::xarray<double>& y) {
    // initialize feature weights
    std::cout << "rows: " << X.shape(0) << ", cols: " << X.shape(1) << std::endl;
    xt::xarray<double> coefs = xt::ones<double>({1, int(X.shape(1))});
    // get baseline pairwise feature distances between samples
    // std::cout << "feature distances. " << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    xt::xarray<double> feature_dists = distance::pairwise_feature_distance(X,
                                                                   metric_, p_);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "feature_dists() took: " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()   << "seconds" << std::endl;                                                     
    // std::cout << "got them distances" << std::endl;
    NCFSOptimizer optimizer(alpha_);
    // set kernel
    KernelMixin kernel(sigma_, lambda_, coefs);
    if (kernel_ == "exponential") {
        ExponentialKernel kernel = ExponentialKernel(sigma_, lambda_, coefs);
    } else {
        GaussianKernel kernel = GaussianKernel(sigma_, lambda_, coefs);
    }
    // construct class adjacency matrix
    xt::xarray<double> class_matrix = format_class_matrix(y);
    // initialize scores
    double objective = 0;
    double loss = std::numeric_limits<float>::infinity();
    int diag_idxs[int(X.shape(0))][2];
    for (int i=0; i < X.shape(0); i++) {
        diag_idxs[i][0] = i;
        diag_idxs[i][1] = i;
    }
    t1 = std::chrono::high_resolution_clock::now();
    while (std::abs(loss) > eta_) {
        auto f_shape = feature_dists.shape();
        // std::cout << "feature distances shape: " << std::endl;
        for (auto& el : f_shape) {std::cout << el << ", "; }
        std::cout << std::endl;
        // convert [1, n_features] weight array to [n_features, 1, 1] array
        auto casted = xt::view(coefs * coefs, 0, xt::all(),
                               xt::newaxis(), xt::newaxis());
        auto c_shape = casted.shape();
        // std::cout << "casted shape: " << std::endl;
        for (auto& el : c_shape) {std::cout << el << ", "; }
        // std::cout << std::endl;
        auto distances = xt::sum(feature_dists * casted, 0);
        // std::cout << "made a distance matrix!" << std::endl;
        auto p_reference = kernel.transform(distances);
        // pseudo counts to avoid dividing by zero in row sums
        p_reference = p_reference + 1e-20;
        for (int i=0; i < p_reference.shape(0); i++) {
            p_reference(i, i) = 0.0;
        }
        auto row_sums = xt::sum(p_reference, 1);
        // std::cout << "row sums: " << row_sums << std::endl;
        auto scale_factors = 1.0 / row_sums;
        p_reference = xt::transpose(xt::transpose(p_reference) * scale_factors);
        auto gradients = kernel.gradients(p_reference, feature_dists,
                                          class_matrix);
        // std::cout << "gradients: " << gradients << std::endl;
        auto weight_sum = xt::sum(coefs * coefs);
        // std::cout << "weights: " << kernel.get_weights() << std::endl;
        // std::cout << "weight sum: " << weight_sum << std::endl;
        double new_objective = (xt::sum(p_reference * class_matrix)
                             - lambda_ * weight_sum)();
        loss = new_objective - objective;
        auto deltas = optimizer.steps(gradients, loss);
        coefs = coefs + deltas;
        kernel.set_weights(coefs);
        objective = new_objective;
        // std::cout << "deltas: " << deltas << std::endl;
    }
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "fitting() took: " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()   << "seconds" << std::endl;                                                     
    return kernel.get_weights();
}
} // end namespace definition
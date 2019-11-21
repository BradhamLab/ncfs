#include <iostream>
#include <eigen3/Eigen/Dense>
#include <bits/stdc++.h>
#include "NCFS.hpp"

using namespace Eigen;
using namespace std;

/**
 * @brief Calculate the L1 distance matrix between all rows in X.
 * 
 * @tparam Derived 
 * @param X A data matrix with rows as samples and features as columns
 * @param dist A (sample x sample) matrix where d_ij will be the distance
 *     between X_i and X_j
 */
template <typename Derived>
void distance_matrix(const DenseBase<Derived>& X, DenseBase<Derived>& dist) {
    for (int i = 0; i < X.rows(); i++) {
        const auto &row = X.row(i);
        #ifdef _OPENMP
        #pragma omp parallel for shared(row)
        #endif
        for (int j = i + 1; j < X.rows(); j++) {
            // dist(i, j) = (X.row(i)- X.row(j)).template lpNorm<1>();
            dist(i, j) = (row - X.row(j)).cwiseAbs().sum();
            dist(j, i) = dist(i, j);
        }
    }
}

template <typename Derived, typename Derived1>
void distance_matrix(const DenseBase<Derived>& X,
                     DenseBase<Derived1>& dist) {
    for (int i = 0; i < X.rows(); i++) {
        for (int j = i + 1; j < X.rows(); j++) {
            dist(i, j) = (X.row(i) - X.row(j)).template lpNorm<1>();
            dist(j, i) = dist(i, j);
        }
    }
}
template <typename Derived, typename Derived1, typename Derived2>
double fit_weights(const MatrixBase<Derived>& X,
                   const ArrayBase<Derived1>& class_matrix,
                   double& objective,
                   MatrixBase<Derived2>& coefs,
                   const double& sigma,
                   const double& lambda,
                   double& alpha) {
    // instantiate vector of weight gradients
    VectorXd gradients = VectorXd::Ones(X.cols());
    // instatiate pseudocounts to avoid division by zero
    VectorXd pseudocounts = VectorXd::Ones(X.cols()).array() * exp(-20);
    // calculate distance between samples
    MatrixXd distance = MatrixXd(X.rows(), X.rows());
    distance_matrix(X, distance);
    // calculate matrix P where p_ij is the probability of selecting sample
    // j as the referenc sample for sample i
    MatrixXd p_reference = (-1 * distance).array().exp();
    // fill diagnol with zeros
    p_reference -= p_reference.diagonal();
    p_reference = p_reference.transpose() \
                * ((p_reference.colwise().sum().cwiseInverse()\
                    + pseudocounts));

    // calculate probability of correct assignment
    VectorXd p_correct = (p_reference.array() * class_matrix).colwise().sum();
    MatrixXd feature_dist = MatrixXd(X.rows(), X.rows());
    // calculate gradients for each feature weight
    for (int l = 0; l < X.cols(); l++) {
        distance_matrix(X.col(l), feature_dist);
        gradients(l) = 2.0 * coefs(l) * ((1.0 / sigma)\
                     * (p_correct * feature_dist.colwise().sum()\
                        - (feature_dist.array() * class_matrix).matrix()\
                                 .colwise().sum()).sum() - lambda);
    }
    // calculate objective function
    double new_objective = (p_reference.array() * class_matrix).sum()\
                         - lambda * coefs.dot(coefs);
    // calculate difference between new objective score and previous
    double loss = new_objective - objective;
    coefs += gradients * alpha;
    if ((objective - new_objective) > 0) {
        alpha *= 1.01;
    } else {
        alpha *= 0.4;
    }
    objective = new_objective;
    return loss;
}

template <typename Derived, typename Derived2>
VectorXd fit(const MatrixBase<Derived>& X,
             const ArrayBase<Derived2>& class_matrix,
             const double& sigma,
             const double& lambda,
             double& alpha,
             const double& eta) {
    double loss = Infinity;
    double objective = 0;
    // initialize weights
    VectorXd coefs = VectorXd::Ones(X.cols());
    while (abs(loss) > eta) {
        loss = fit_weights(X, class_matrix, objective, coefs, sigma, lambda,
                           alpha);
    }

    return(coefs);
}

int main() {
    MatrixXd m = MatrixXd::Random(1000, 100);
    MatrixXd d = MatrixXd(1000, 1000);
    VectorXd v = VectorXd::Ones(100);
    time_t start, end;
    time(&start);
    distance_matrix(m * v, d);
    time(&end);
    // Calculating total time taken by the program. 
    double time_taken = double(end - start); 
    cout << "Time taken by program is : " << fixed 
         << time_taken << setprecision(5); 
    cout << " sec " << endl; 
    return 0; 
}

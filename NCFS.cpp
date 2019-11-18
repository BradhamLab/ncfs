#include <iostream>
#include <eigen3/Eigen/Dense>
#include <bits/stdc++.h>

using namespace Eigen;
using namespace std;

MatrixXd distance_matrix(MatrixXd X) {
    MatrixXd dist = MatrixXd::Zero(X.rows(), X.rows());
    for (int i = 0; i < X.rows(); i++) {
        for (int j = i + 1; j < X.rows(); j++) {
            dist(i, j) = (X.row(i) - X.row(j)).lpNorm<1>();
            dist(j, i) = dist(i, j);
        }
    }
    return dist;
};

double __fit(MatrixXd &X, ArrayXd &class_matrix, double &objective,
           VectorXd &coefs, double &sigma, double &lambda, double &alpha) {
    // instantiate vector of weight gradients
    VectorXd gradients = VectorXd::Ones(X.cols());
    // instatiate pseudocounts to avoid division by zero
    VectorXd pseudocounts = VectorXd::Ones(X.cols()).array() * exp(-20);
    // calculate matrix P where p_ij is the probability of selecting sample
    // j as the referenc sample for sample i
    MatrixXd p_reference = (-1 * distance_matrix(X * coefs)).array().exp();
    // fill diagnol with zeros
    p_reference -= p_reference.diagonal();
    p_reference = p_reference.transpose() \
                * ((p_reference.colwise().sum().cwiseInverse()\
                    + pseudocounts));

    // calculate probability of correct assignment
    VectorXd p_correct = (p_reference.array() * class_matrix).colwise().sum();
    // calculate gradients for each feature weight
    for (int l = 0; l < X.cols(); l++) {
        MatrixXd feature_dist = distance_matrix(X.col(l));
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

VectorXd fit(MatrixXd X, ArrayXd class_matrix, double sigma, double lambda,
             double alpha, double eta) {
    double loss = Infinity;
    double objective = 0;
    // initialize weights
    VectorXd coefs = VectorXd::Ones(X.cols());
    while (abs(loss) > eta) {
        __fit(X, class_matrix, objective, coefs, sigma, lambda, alpha);
    }

    return(coefs);
};

int main() {
    MatrixXd m = MatrixXd::Random(10, 10);
    VectorXd v = VectorXd::Ones(10);
    time_t start, end;
    time(&start);
    distance_matrix(m * v);
    time(&end);
    // Calculating total time taken by the program. 
    double time_taken = double(end - start); 
    cout << "Time taken by program is : " << fixed 
         << time_taken << setprecision(5); 
    cout << " sec " << endl; 
    return 0; 
}

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <bits/stdc++.h>

using namespace Eigen;
using namespace std;

MatrixXd distance_matrix(MatrixXd X) {
    MatrixXd dist = MatrixXd::Zero(X.rows(), X.rows());
    for (int i = 0; i < X.rows(); i++) {
        for (int j = i + 1; j < X.rows(); j++) {
            // cout << (X.row(i) - X.row(j)).lpNorm<1>();
            dist(i, j) = (X.row(i) - X.row(j)).lpNorm<1>();
            dist(j, i) = dist(i, j);
        }
    }
    return dist;
};

VectorXd fit(MatrixXd X, VectorXd y) {
    // Construct class adjacency matrix
    MatrixXd class_matrix = MatrixXd::Zero(y.size(), y.size());
    for (int i = 0; i < y.size(); i++) {
        for (int j = i + 1; j < y.size(); j++) {
            if (y(i) == y(j)) {
                class_matrix(i, j) = 1;
                class_matrix(j, i) = 1;
            }
        }
    }
    // initialize weights
    VectorXd coef_ = VectorXd::Ones(X.cols());
    VectorXd gradients = VectorXd::Ones(X.cols());
    VectorXd pseudocounts = VectorXd::Ones(X.cols()).array() * exp(-20);
    MatrixXd p_reference = (-1 * distance_matrix(X * coef_)).array().exp();
    double sigma = 1;
    double reg = 1;
    // fill diagnol with zeros
    p_reference -= p_reference.diagonal();
    p_reference = p_reference.transpose() \
                * ((p_reference.colwise().sum().cwiseInverse()\
                    + pseudocounts));

    //
    VectorXd p_correct = (p_reference.array() * class_matrix.array()).colwise().sum();
    for (int l = 0; l < X.cols(); l++) {
        MatrixXd feature_dist = distance_matrix(X.col(l));
        gradients(l) = 2.0 * coef_(l) * ((1.0 / sigma)\
                     * p_correct * feature_dist.colwise().sum()\
                     - (feature_dist.array() * class_matrix.array()).colwise().sum());

    }

    return(coef_);
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
};

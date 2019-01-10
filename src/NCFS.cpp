#include <iostream>
#include <math.h>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xtensor.hpp"

xt::xarray<double> kernel_distance(xt::xarray<double> X,
                                   xt::xarray<double> weights,
                                   double sigma) {
    auto dims = X.shape();
    int n_samples = dims[0];
    int n_features = dims[1];
    xt::xarray<double> distance = xt::zeros<double>({n_samples, n_samples});
    for (int i=0; i < n_samples; i++) {
      for (int j=0; j < n_samples; j++) {
        if (i == j) {
          distance(i, j) = 0;
        } else {
          auto dist = xt::square(xt::view(X, i) - xt::view(X, j));
          xt::xarray<double> val = xt::exp(-1 * xt::sum(dist * weights) / sigma);
          distance(i, j) = val(0, 0, 0);
        }
      }
    }
    return distance; 
}

xt::xarray<double> fit(xt::xarray<double> X, xt::xarray<double> y,
                       double alpha, double lambda, double sigma, double eta) {

    auto dims = X.shape();
    int n_samples = dims[0];
    int n_features = dims[1];
    xt::xarray<double> weights = xt::ones<double>({n_features, 1});
    xt::xarray<double> deltas = xt::ones<double>({n_features, 1});
    xt::xarray<double> class_matrix = xt::zeros<double>({n_samples, n_samples});
    float step_size = alpha;
    std::cout << y << std::endl;

    for (int i=0; i < n_samples; i++) {
      for (int j=0; j < n_samples; j++) {
        if (y[i] == y[j]) {
          class_matrix(i, j) = 1.0;
        }
      }
    }
    float loss = 1000000;
    float past_objective = 0;

    while (abs(loss) > eta) {
        xt::xarray<double> p_reference = kernel_distance(X, weights, sigma);
        xt::xarray<double> marginal = xt::sum(p_reference, axis=0);
        // add pseudo counts here if necessary
        p_reference = p_reference * (1 / marginal);
        xt::xarray<double> p_correct = xt::sum(p_reference * class_matrix,
                                               axis=0);
        for (int l=0; l < n_features; l++) {
          auto feature_vec = xt::view(X, xt::all(), l)
          
        }
    }
    return class_matrix;
}



int main(int argc, char* argv[])
{
    xt::xarray<double> arr1
      {{1.0, 2.0, 3.0, 5.0},
       {2.0, 5.0, 7.0, 5.0},
       {2.0, 5.0, 7.0, 5.0}};

    xt::xarray<double> arr2
      {1.0, 1.0, 1.0, 1.0};

    xt::xarray<double> res = kernel_distance(arr1, arr2, 1);
    std::cout << res << std::endl;

    // xt::xarray<double> res = fit(arr1, arr2, 0.1, 1, 1, 0.001);
    double val = arr1[0, 0];
    auto dims = arr1.shape();


    return 0;
}
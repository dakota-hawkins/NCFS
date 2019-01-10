#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xtensor.hpp"

xt::xtensor<double> fit(xt::xtensor<double> X, xt::xtensor<double> y,
                       double alpha, double lambda, double eta) {

    int n_samples, n_features = X.shape();
    xt::xtensor<double> weights = xt::ones<double>(n_features);
    xt::xtensor<double> deltas = xt::ones<double>(n_features);
    xt::xtensor<double> class_matrix = xt::zeros<double>({n_samples, n_samples});


    return weights
}

int main(int argc, char* argv[])
{
    xt::xarray<double> arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};

    xt::xarray<double> arr2
      {5.0, 6.0, 7.0};

    xt::xarray<double> res = xt::view(arr1, 1) + arr2;
    double val = arr{0, 0};

    std::cout << val;

    return 0;
}
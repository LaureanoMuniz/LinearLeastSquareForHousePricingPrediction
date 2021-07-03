#include <algorithm>
//#include <chrono>
#include <pybind11/pybind11.h>
#include <Eigen/SVD>
#include <iostream>
#include <exception>
#include "linear_regression.h"

using namespace std;
namespace py=pybind11;


const double eps = 1e-9;


LinearRegression::LinearRegression() {}

void LinearRegression::fit(Matrix A, Matrix b) {
    Eigen::JacobiSVD<Matrix> svd(A, Eigen::ComputeThinV | Eigen::ComputeThinU);
    auto singular_values = svd.singularValues();
    auto V = svd.matrixV();
    auto U = svd.matrixU();
    b = U.transpose() * b;
    Eigen::Index range = 0;
    for(Eigen::Index i = 0; i < singular_values.size(); ++i) {
        auto x = singular_values(i);
        if(x > eps) {
            range++;
        }
    }
    Matrix y = Matrix::Zero(A.cols(), b.cols());
    for(Eigen::Index i = 0; i < b.cols(); i++) { 
        for(Eigen::Index j = 0; j < range; j++) {
            y(j, i) = b(j, i) / singular_values(j);
        }
    }
    this->x = V * y;
    return;
}

Matrix LinearRegression::predict(Matrix C) {
    return C * this->x;
}


Matrix LinearRegression::get_X(){
    return this->x;
}
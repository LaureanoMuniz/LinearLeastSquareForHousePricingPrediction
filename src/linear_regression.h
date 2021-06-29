#pragma once

#include "types.h"

class LinearRegression {
public:
    LinearRegression();

    void fit(Matrix A, Matrix b);

    Matrix predict(Matrix C);

    Matrix get_X();
private:
    Matrix x;
};

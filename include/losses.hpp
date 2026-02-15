#pragma once

#include "matrix.hpp"

namespace nn {

enum class LossType {
    MSE,
    BinaryCrossEntropy
};

double lossValue(LossType lossType, const Matrix& predictions, const Matrix& targets);
Matrix lossGradient(LossType lossType, const Matrix& predictions, const Matrix& targets);

} // namespace nn

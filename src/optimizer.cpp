#include "optimizer.hpp"

namespace nn {

SGD::SGD(double learningRate) : learningRate_(learningRate) {}

void SGD::update(Matrix& param, const Matrix& grad) {
    param -= grad * learningRate_;
}

} // namespace nn

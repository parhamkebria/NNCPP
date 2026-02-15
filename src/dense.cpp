#include "dense.hpp"

#include <cmath>
#include <stdexcept>

#include "optimizer.hpp"

namespace nn {

DenseLayer::DenseLayer(std::size_t inputSize, std::size_t outputSize)
    : weights_(Matrix::random(inputSize, outputSize,
                            -std::sqrt(2.0 / static_cast<double>(inputSize)),
                            std::sqrt(2.0 / static_cast<double>(inputSize)))),
    biases_(Matrix::zeros(1, outputSize)),
    gradWeights_(Matrix::zeros(inputSize, outputSize)),
    gradBiases_(Matrix::zeros(1, outputSize)) {}

Matrix DenseLayer::forward(const Matrix& input) {
    if (input.cols() != weights_.rows()) {
        throw std::invalid_argument("Input feature count does not match layer input size");
    }

    lastInput_ = input;
    Matrix output = input.dot(weights_);
    output.addRowVectorInPlace(biases_);
    return output;
}

Matrix DenseLayer::backward(const Matrix& gradOutput) {
    if (gradOutput.cols() != weights_.cols()) {
        throw std::invalid_argument("gradOutput width does not match layer output size");
    }

    const double batchSize = static_cast<double>(lastInput_.rows());

    gradWeights_ = lastInput_.transpose().dot(gradOutput) / batchSize;
    gradBiases_ = gradOutput.sumRows() / batchSize;

    Matrix gradInput = gradOutput.dot(weights_.transpose());
    return gradInput;
}

void DenseLayer::updateParams(Optimizer& optimizer) {
    optimizer.update(weights_, gradWeights_);
    optimizer.update(biases_, gradBiases_);
}

} // namespace nn

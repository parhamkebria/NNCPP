#pragma once

#include "layer.hpp"

namespace nn {

class DenseLayer final : public Layer {
public:
    DenseLayer(std::size_t inputSize, std::size_t outputSize);

    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradOutput) override;
    void updateParams(Optimizer& optimizer) override;

private:
    Matrix weights_;
    Matrix biases_;

    Matrix gradWeights_;
    Matrix gradBiases_;

    Matrix lastInput_;
};

} // namespace nn

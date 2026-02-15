#pragma once

#include "layer.hpp"

namespace nn {

enum class ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Linear
};

Matrix activate(const Matrix& input, ActivationType type);
Matrix activationDerivative(const Matrix& input, const Matrix& output, ActivationType type);

class ActivationLayer final : public Layer {
public:
    explicit ActivationLayer(ActivationType type);

    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradOutput) override;
    void updateParams(Optimizer& optimizer) override;

private:
    ActivationType type_;
    Matrix lastInput_;
    Matrix lastOutput_;
};

} // namespace nn

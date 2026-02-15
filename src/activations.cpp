#include "nn/activations.hpp"

#include <cmath>
#include <stdexcept>

namespace nn {

namespace {
double applyActivation(double x, ActivationType type) {
    switch (type) {
        case ActivationType::ReLU:
            return x > 0.0 ? x : 0.0;
        case ActivationType::Sigmoid:
            return 1.0 / (1.0 + std::exp(-x));
        case ActivationType::Tanh:
            return std::tanh(x);
        case ActivationType::Linear:
            return x;
        default:
            throw std::invalid_argument("Unsupported activation type");
    }
}

double activationPrime(double x, double y, ActivationType type) {
    switch (type) {
        case ActivationType::ReLU:
            return x > 0.0 ? 1.0 : 0.0;
        case ActivationType::Sigmoid:
            return y * (1.0 - y);
        case ActivationType::Tanh:
            return 1.0 - y * y;
        case ActivationType::Linear:
            return 1.0;
        default:
            throw std::invalid_argument("Unsupported activation type");
    }
}
} // namespace

Matrix activate(const Matrix& input, ActivationType type) {
    Matrix output(input.rows(), input.cols(), 0.0);
    for (std::size_t r = 0; r < input.rows(); ++r) {
        for (std::size_t c = 0; c < input.cols(); ++c) {
            output(r, c) = applyActivation(input(r, c), type);
        }
    }
    return output;
}

Matrix activationDerivative(const Matrix& input, const Matrix& output, ActivationType type) {
    if (input.rows() != output.rows() || input.cols() != output.cols()) {
        throw std::invalid_argument("Input and output dimensions must match for activation derivative");
    }

    Matrix derivative(input.rows(), input.cols(), 0.0);
    for (std::size_t r = 0; r < input.rows(); ++r) {
        for (std::size_t c = 0; c < input.cols(); ++c) {
            derivative(r, c) = activationPrime(input(r, c), output(r, c), type);
        }
    }
    return derivative;
}

ActivationLayer::ActivationLayer(ActivationType type) : type_(type) {}

Matrix ActivationLayer::forward(const Matrix& input) {
    lastInput_ = input;
    lastOutput_ = activate(input, type_);
    return lastOutput_;
}

Matrix ActivationLayer::backward(const Matrix& gradOutput) {
    Matrix deriv = activationDerivative(lastInput_, lastOutput_, type_);
    return gradOutput.hadamard(deriv);
}

void ActivationLayer::updateParams(Optimizer&) {
}

} // namespace nn

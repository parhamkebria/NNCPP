#include "nn/losses.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nn {

namespace {
constexpr double kEps = 1e-12;
}

double lossValue(LossType lossType, const Matrix& predictions, const Matrix& targets) {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }

    const std::size_t n = predictions.rows() * predictions.cols();
    if (n == 0) {
        throw std::invalid_argument("Predictions and targets cannot be empty");
    }

    double loss = 0.0;

    switch (lossType) {
        case LossType::MSE:
            for (std::size_t r = 0; r < predictions.rows(); ++r) {
                for (std::size_t c = 0; c < predictions.cols(); ++c) {
                    const double diff = predictions(r, c) - targets(r, c);
                    loss += diff * diff;
                }
            }
            return loss / static_cast<double>(n);

        case LossType::BinaryCrossEntropy:
            for (std::size_t r = 0; r < predictions.rows(); ++r) {
                for (std::size_t c = 0; c < predictions.cols(); ++c) {
                    const double y = targets(r, c);
                    const double p = std::clamp(predictions(r, c), kEps, 1.0 - kEps);
                    loss += -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
                }
            }
            return loss / static_cast<double>(n);

        default:
            throw std::invalid_argument("Unsupported loss type");
    }
}

Matrix lossGradient(LossType lossType, const Matrix& predictions, const Matrix& targets) {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }

    const std::size_t n = predictions.rows() * predictions.cols();
    if (n == 0) {
        throw std::invalid_argument("Predictions and targets cannot be empty");
    }

    Matrix grad(predictions.rows(), predictions.cols(), 0.0);

    switch (lossType) {
        case LossType::MSE:
            for (std::size_t r = 0; r < predictions.rows(); ++r) {
                for (std::size_t c = 0; c < predictions.cols(); ++c) {
                    grad(r, c) = 2.0 * (predictions(r, c) - targets(r, c)) / static_cast<double>(n);
                }
            }
            return grad;

        case LossType::BinaryCrossEntropy:
            for (std::size_t r = 0; r < predictions.rows(); ++r) {
                for (std::size_t c = 0; c < predictions.cols(); ++c) {
                    const double y = targets(r, c);
                    const double p = std::clamp(predictions(r, c), kEps, 1.0 - kEps);
                    grad(r, c) = (p - y) / ((p * (1.0 - p)) * static_cast<double>(n));
                }
            }
            return grad;

        default:
            throw std::invalid_argument("Unsupported loss type");
    }
}

} // namespace nn

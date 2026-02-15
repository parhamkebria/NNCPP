#pragma once

#include "matrix.hpp"

namespace nn {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update(Matrix& param, const Matrix& grad) = 0;
};

class SGD final : public Optimizer {
public:
    explicit SGD(double learningRate);
    void update(Matrix& param, const Matrix& grad) override;

private:
    double learningRate_;
};

} // namespace nn

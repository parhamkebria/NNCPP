#pragma once

#include "matrix.hpp"

namespace nn {

class Optimizer;

class Layer {
public:
    virtual ~Layer() = default;

    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& gradOutput) = 0;
    virtual void updateParams(Optimizer& optimizer) = 0;
};

} // namespace nn

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "layer.hpp"
#include "losses.hpp"
#include "optimizer.hpp"

namespace nn {

class NeuralNetwork {
public:
    NeuralNetwork();

    void addLayer(std::unique_ptr<Layer> layer);

    Matrix forward(const Matrix& input);
    Matrix predict(const Matrix& input);

    double train(const Matrix& inputs,
                const Matrix& targets,
                std::size_t epochs,
                LossType lossType,
                Optimizer& optimizer,
                bool verbose = true);

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

} // namespace nn

#include "neural_network.hpp"

#include <iostream>
#include <stdexcept>

namespace nn {

NeuralNetwork::NeuralNetwork() = default;

void NeuralNetwork::addLayer(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
}

Matrix NeuralNetwork::forward(const Matrix& input) {
    Matrix output = input;
    for (const auto& layer : layers_) {
        output = layer->forward(output);
    }
    return output;
}

Matrix NeuralNetwork::predict(const Matrix& input) {
    return forward(input);
}

double NeuralNetwork::train(const Matrix& inputs,
                            const Matrix& targets,
                            std::size_t epochs,
                            LossType lossType,
                            Optimizer& optimizer,
                            bool verbose) {
    if (layers_.empty()) {
        throw std::invalid_argument("Cannot train network with no layers");
    }
    if (inputs.rows() != targets.rows()) {
        throw std::invalid_argument("Inputs and targets must have the same number of rows");
    }

    double currentLoss = 0.0;

    for (std::size_t epoch = 1; epoch <= epochs; ++epoch) {
        Matrix predictions = forward(inputs);

        currentLoss = lossValue(lossType, predictions, targets);
        Matrix grad = lossGradient(lossType, predictions, targets);

        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            grad = (*it)->backward(grad);
        }

        for (auto& layer : layers_) {
            layer->updateParams(optimizer);
        }

        if (verbose && (epoch == 1 || epoch % 100 == 0 || epoch == epochs)) {
            std::cout << "Epoch " << epoch << "/" << epochs
                      << " - Loss: " << currentLoss << '\n';
        }
    }

    return currentLoss;
}

} // namespace nn

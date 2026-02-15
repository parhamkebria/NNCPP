#include <iostream>
#include <memory>

#include "activations.hpp"
#include "dense.hpp"
#include "neural_network.hpp"
#include "optimizer.hpp"

int main() {
    nn::Matrix x{{0.0, 0.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 1.0}};

    nn::Matrix y{{0.0},
                {1.0},
                {1.0},
                {0.0}};

    nn::NeuralNetwork network;
    network.addLayer(std::make_unique<nn::DenseLayer>(2, 4));
    network.addLayer(std::make_unique<nn::ActivationLayer>(nn::ActivationType::Tanh));
    network.addLayer(std::make_unique<nn::DenseLayer>(4, 1));
    network.addLayer(std::make_unique<nn::ActivationLayer>(nn::ActivationType::Sigmoid));

    nn::SGD optimizer(0.1);

    network.train(x, y, 5000, nn::LossType::BinaryCrossEntropy, optimizer, true);

    nn::Matrix pred = network.predict(x);
    std::cout << "\nPredictions after training:\n";
    for (std::size_t i = 0; i < pred.rows(); ++i) {
        std::cout << x(i, 0) << " XOR " << x(i, 1) << " = " << pred(i, 0) << '\n';
    }

    return 0;
}

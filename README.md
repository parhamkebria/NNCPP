# NNCPP

Minimal C++ neural network library with a small XOR training demo.

## Features

- Core `Matrix` type with basic linear algebra operations.
- Layer interface with `DenseLayer` and `ActivationLayer` implementations.
- Activations: `ReLU`, `Sigmoid`, `Tanh`, `Linear`.
- Losses: `MSE`, `BinaryCrossEntropy`.
- Optimizer: `SGD`.
- `NeuralNetwork` class with forward, predict, and training loops.

## Build

Requires CMake (3.16+) and a C++17 compiler.

```bash
cmake -S . -B build
cmake --build build
```

## Run demo

```bash
./build/neural_net_demo
```

The demo trains a small network on the XOR dataset and prints predictions.

## Project layout

- `include/`: public headers
- `src/`: implementation files
- `main.cpp`: XOR training example

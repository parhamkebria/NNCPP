#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace nn {

class Matrix {
public:
    Matrix();
    Matrix(std::size_t rows, std::size_t cols, double value = 0.0);
    Matrix(std::initializer_list<std::initializer_list<double>> values);

    std::size_t rows() const;
    std::size_t cols() const;

    double& operator()(std::size_t row, std::size_t col);
    double operator()(std::size_t row, std::size_t col) const;

    const std::vector<double>& data() const;
    std::vector<double>& data();

    static Matrix random(std::size_t rows, std::size_t cols, double min = -1.0, double max = 1.0);
    static Matrix zeros(std::size_t rows, std::size_t cols);

    Matrix transpose() const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar);

    Matrix hadamard(const Matrix& other) const;
    Matrix dot(const Matrix& other) const;
    Matrix sumRows() const;

    void addRowVectorInPlace(const Matrix& rowVector);

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;
};

} // namespace nn

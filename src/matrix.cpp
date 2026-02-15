#include "matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace nn {

Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, double value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> values) {
    rows_ = values.size();
    cols_ = rows_ == 0 ? 0 : values.begin()->size();

    data_.reserve(rows_ * cols_);
    for (const auto& row : values) {
        if (row.size() != cols_) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
        data_.insert(data_.end(), row.begin(), row.end());
    }
}

std::size_t Matrix::rows() const { return rows_; }
std::size_t Matrix::cols() const { return cols_; }

double& Matrix::operator()(std::size_t row, std::size_t col) {
    return data_.at(row * cols_ + col);
}

double Matrix::operator()(std::size_t row, std::size_t col) const {
    return data_.at(row * cols_ + col);
}

const std::vector<double>& Matrix::data() const { return data_; }
std::vector<double>& Matrix::data() { return data_; }

Matrix Matrix::random(std::size_t rows, std::size_t cols, double min, double max) {
    if (min > max) {
        throw std::invalid_argument("min must be <= max");
    }

    Matrix result(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);

    for (double& value : result.data_) {
        value = dist(gen);
    }
    return result;
}

Matrix Matrix::zeros(std::size_t rows, std::size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (std::size_t r = 0; r < rows_; ++r) {
        for (std::size_t c = 0; c < cols_; ++c) {
            result(c, r) = (*this)(r, c);
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    if (std::fabs(scalar) < std::numeric_limits<double>::epsilon()) {
        throw std::invalid_argument("Division by zero scalar");
    }

    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] / scalar;
    }
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for addition assignment");
    }

    for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction assignment");
    }

    for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    for (double& value : data_) {
        value *= scalar;
    }
    return *this;
}

Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for Hadamard product");
    }

    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Matrix Matrix::dot(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Invalid matrix dimensions for multiplication");
    }

    Matrix result(rows_, other.cols_, 0.0);

    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t k = 0; k < cols_; ++k) {
            const double a = (*this)(i, k);
            for (std::size_t j = 0; j < other.cols_; ++j) {
                result(i, j) += a * other(k, j);
            }
        }
    }
    return result;
}

Matrix Matrix::sumRows() const {
    Matrix result(1, cols_, 0.0);
    for (std::size_t r = 0; r < rows_; ++r) {
        for (std::size_t c = 0; c < cols_; ++c) {
            result(0, c) += (*this)(r, c);
        }
    }
    return result;
}

void Matrix::addRowVectorInPlace(const Matrix& rowVector) {
    if (rowVector.rows_ != 1 || rowVector.cols_ != cols_) {
        throw std::invalid_argument("Row vector dimensions must be (1, cols)");
    }

    for (std::size_t r = 0; r < rows_; ++r) {
        for (std::size_t c = 0; c < cols_; ++c) {
            (*this)(r, c) += rowVector(0, c);
        }
    }
}

} // namespace nn

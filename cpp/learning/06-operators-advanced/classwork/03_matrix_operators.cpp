/*
 * MODULE 6: Operator Overloading
 * File: 03_matrix_operators.cpp
 *
 * TOPIC: Matrix Class with Operators for GPU Graphics
 *
 * COMPILE: g++ -std=c++17 -o matrix_operators 03_matrix_operators.cpp
 */

#include <iostream>
#include <cmath>
#include <iomanip>

class Mat4 {
public:
    float m[16];  // Column-major order (like OpenGL)

    Mat4() { identity(); }

    void identity() {
        for (int i = 0; i < 16; ++i) m[i] = 0;
        m[0] = m[5] = m[10] = m[15] = 1.0f;
    }

    // Access operator: mat(row, col)
    float& operator()(int row, int col) {
        return m[col * 4 + row];  // Column-major
    }

    const float& operator()(int row, int col) const {
        return m[col * 4 + row];
    }

    // Matrix multiplication
    Mat4 operator*(const Mat4& other) const {
        Mat4 result;
        for (int i = 0; i < 16; ++i) result.m[i] = 0;

        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                for (int k = 0; k < 4; ++k) {
                    result(row, col) += (*this)(row, k) * other(k, col);
                }
            }
        }
        return result;
    }

    // Create transformation matrices
    static Mat4 translation(float x, float y, float z) {
        Mat4 mat;
        mat(0, 3) = x;
        mat(1, 3) = y;
        mat(2, 3) = z;
        return mat;
    }

    static Mat4 scale(float x, float y, float z) {
        Mat4 mat;
        mat(0, 0) = x;
        mat(1, 1) = y;
        mat(2, 2) = z;
        return mat;
    }

    static Mat4 rotationZ(float angle) {
        Mat4 mat;
        float c = std::cos(angle);
        float s = std::sin(angle);
        mat(0, 0) = c;  mat(0, 1) = -s;
        mat(1, 0) = s;  mat(1, 1) = c;
        return mat;
    }

    void print() const {
        std::cout << std::fixed << std::setprecision(2);
        for (int row = 0; row < 4; ++row) {
            std::cout << "| ";
            for (int col = 0; col < 4; ++col) {
                std::cout << std::setw(6) << (*this)(row, col) << " ";
            }
            std::cout << "|\n";
        }
    }
};

int main() {
    std::cout << "=== Matrix Operators for GPU Transform===\n\n";

    Mat4 trans = Mat4::translation(5, 0, 0);
    Mat4 rot = Mat4::rotationZ(3.14159f / 4);  // 45 degrees
    Mat4 scl = Mat4::scale(2, 2, 2);

    std::cout << "Translation matrix:\n";
    trans.print();

    std::cout << "\nRotation matrix (45 deg):\n";
    rot.print();

    std::cout << "\nCombined: Scale * Rotate * Translate\n";
    Mat4 combined = scl * rot * trans;
    combined.print();

    std::cout << "\nGPU CONNECTION: Identical to glm::mat4 operations!\n";
    return 0;
}

/*
 * TRY THIS:
 * 1. Add perspective projection matrix
 * 2. Implement matrix inversion
 * 3. Add lookAt camera matrix
 * 4. Implement matrix-vector multiplication
 * 5. Add rotation matrices for X and Y axes
 */

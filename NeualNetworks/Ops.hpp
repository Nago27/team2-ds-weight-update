#pragma once
#include "Matrix.hpp"

namespace vsnn {
    class Ops {
    public:
        // ReLU 활성화 함수 (Forward)
        static void ReLUForward(const Matrix& X, Matrix& Y) {
            Y = X.cwiseMax(0.0f);
        }

        // ReLU 활성화 함수 (Backward)
        static void ReLUBackward(const Matrix& X, const Matrix& dY, Matrix& dX) {
            dX = (X.array() > 0.0f).select(dY, 0.0f);
        }

        // Softmax 함수 (행렬 전체에 대해 행별로 적용)
        static void Softmax(const Matrix& logits, Matrix& probs) {
            if (probs.rows() != logits.rows() || probs.cols() != logits.cols()) {
                probs.resize(logits.rows(), logits.cols());
            }

            for (Eigen::Index i = 0; i < logits.rows(); ++i) {
                float max_val = logits.row(i).maxCoeff();
                Matrix exp_row = (logits.row(i).array() - max_val).exp();
                float sum = exp_row.sum();
                if (sum > 0) {
                    probs.row(i) = exp_row / sum;
                }
                else {
                    probs.row(i).setConstant(1.0f / logits.cols());
                }
            }
        }
    };
}
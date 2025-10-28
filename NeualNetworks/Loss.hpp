#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "Matrix.hpp"
#include "Ops.hpp"

namespace vsnn {
    class SoftmaxCrossEntropy {
    private:
        Matrix probs_;
    public:
        float Forward(const Matrix& logits, const std::vector<int>& y) {
            const auto N = logits.rows();
            if (N == 0) return 0.0f;

            Ops::Softmax(logits, probs_);

            float loss = 0.0f;
            for (Eigen::Index n = 0; n < N; ++n) {
                const int target_class = y[n];
                const float p = std::max(1e-12f, probs_(n, target_class));
                loss += -log(p);
            }
            return loss / static_cast<float>(N);
        }

        void Backward(const std::vector<int>& y, Matrix& dLogits) {
            const auto N = probs_.rows();
            if (N == 0) return;

            dLogits = probs_;
            for (Eigen::Index n = 0; n < N; ++n) {
                dLogits(n, y[n]) -= 1.0f;
            }
            dLogits /= static_cast<float>(N);
        }
    };
}
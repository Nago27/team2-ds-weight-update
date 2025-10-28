// =============================
// include/vsnn/Perceptron.hpp (optional demo)
// =============================
#pragma once
#include <vector>
#include "Matrix.hpp"

namespace vsnn {
    class PerceptronBinary {
    private:
        Matrix W_; // 1 x D
        float bias_ = 0.0f;
        static int Sign(float v) { return v >= 0.0f ? 1 : -1; }
    public:
        explicit PerceptronBinary(int dim) : W_(1, dim) { W_.setZero(); } // .Fill(0.0f) -> .setZero()

        void FitEpoch(const Matrix& X, const std::vector<int>& y) {
            const int N = X.rows();
            for (int n = 0; n < N; ++n) {
                // 💡 최적화: 수동 내적 계산을 Eigen의 .dot() 함수로 대체
                float s = bias_ + W_.row(0).dot(X.row(n));

                const int target = (y[n] == 0) ? -1 : 1;
                if (Sign(s) != target) {
                    // 💡 최적화: 가중치 업데이트 루프를 벡터 연산으로 대체
                    W_.row(0) += X.row(n) * static_cast<float>(target);
                    bias_ += static_cast<float>(target);
                }
            }
        }

        int PredictOne(const float* x, int D) const {
            // 💡 최적화: C-스타일 배열 x를 Eigen::Map을 이용해 복사 없이 Eigen 벡터처럼 사용
            float s = bias_ + W_.row(0).dot(Eigen::Map<const Matrix>(x, 1, D));
            return (s >= 0.0f) ? 1 : 0;
        }
    };
    // PerceptronOVR 클래스는 구조적 변경이 필요 없어 그대로 유지합니다.
    class PerceptronOVR { /* ... */ };
}
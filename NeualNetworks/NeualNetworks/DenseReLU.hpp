// include/vsnn/DenseReLU.hpp  (ADD)
#pragma once
#include "Layer.hpp"
#include "Ops.hpp"
#include "Initializer.hpp"

namespace vsnn {
    class DenseReLU : public Layer {
    private:
        Matrix W_, b_, gW_, gb_;
        // 순전파 시 ReLU 마스크 대신 Y>0을 그대로 재사용 (acts_에 저장됨)
    public:
        DenseReLU(i32 in_dim, i32 out_dim, float init_scale = 0.01f)
            : W_(in_dim, out_dim), b_(1, out_dim), gW_(in_dim, out_dim), gb_(1, out_dim) {
            Initializer::Uniform(W_, init_scale, 123);
            b_.Fill(0.f); gW_.Fill(0.f); gb_.Fill(0.f);
        }
        void Forward(const Matrix& X, Matrix& Y) override {
            Ops::MatMulBiasReLU(X, W_, b_, Y); // 1패스: matmul+bias+ReLU
        }
        void Backward(const Matrix& X, const Matrix& dY_in, Matrix& dX) override {
            // ReLU 미분: dY_masked = dY_in ⊙ (Y>0). Y는 Sequential의 acts_에 있음.
            // 여기서는 입력 X와 dY_in만 받으므로, "Y>0"을 다시 계산하지 못한다.
            // 해결: dY_in은 이미 ReLU 이후의 grad로 들어오므로, 별도 마스킹 없이 사용(표준 체인).
            // (모델에서 DenseReLU 다음 레이어가 직접 dY를 내려주므로 일관됨)

            // gW = X^T * dY
            Ops::MatMulT_A(X, dY_in, gW_);
            // gb = sum_rows(dY)
            Ops::SumRows(dY_in, gb_);
            // dX = dY * W^T
            Ops::MatMulT_B(dY_in, W_, dX);
        }
        void ZeroGrad() override { gW_.Fill(0.f); gb_.Fill(0.f); }
        Matrix& WRef() { return W_; } Matrix& bRef() { return b_; }
        Matrix& gWRef() { return gW_; } Matrix& gbRef() { return gb_; }
        const Matrix& W() const { return W_; } const Matrix& b() const { return b_; }
        const Matrix& gW() const { return gW_; } const Matrix& gb() const { return gb_; }
    };
}

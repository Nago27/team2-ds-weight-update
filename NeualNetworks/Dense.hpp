#pragma once
#include "Layer.hpp"
#include "Initializer.hpp"

namespace vsnn {
    class Dense : public Layer {
    private:
        Matrix W_, b_;
        Matrix gW_, gb_;

    public:
        Dense(i32 in_dim, i32 out_dim, float init_scale = 0.01f)
            : W_(in_dim, out_dim), b_(1, out_dim), gW_(in_dim, out_dim), gb_(1, out_dim) {
            Initializer::Uniform(W_, init_scale, 123);
            b_.setZero();
            gW_.setZero();
            gb_.setZero();
        }

        void Forward(const Matrix& X, Matrix& Y) override {
            Y = X * W_;
            Y.rowwise() += b_.row(0);
        }

        void Backward(const Matrix& X, const Matrix& dY, Matrix& dX) override {
            gW_ = X.transpose() * dY;
            gb_ = dY.colwise().sum();
            dX = dY * W_.transpose();
        }

        void ZeroGrad() override {
            gW_.setZero();
            gb_.setZero();
        }

        void Step(float) override {}

        Matrix& WRef() { return W_; }
        Matrix& bRef() { return b_; }
        Matrix& gWRef() { return gW_; }
        Matrix& gbRef() { return gb_; }
    };
}
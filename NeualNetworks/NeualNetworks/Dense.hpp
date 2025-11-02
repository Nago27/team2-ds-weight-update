// =============================
// include/vsnn/Dense.hpp
// =============================
#pragma once
#include "Layer.hpp"
#include "Ops.hpp"
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
			b_.Fill(0.0f); gW_.Fill(0.0f); gb_.Fill(0.0f);
		}

		// Dense.hpp — 교체: Forward
		void Forward(const Matrix& X, Matrix& Y) override {
			// Y = X * W + b  (루프 융합)
			Ops::MatMulBias(X, W_, b_, Y);
		}

		// Dense.hpp — 교체: Backward
		void Backward(const Matrix& X, const Matrix& dY, Matrix& dX) override {
			// gW = X^T * dY  (outer-product 누적 커널)
			Ops::MatMulT_A(X, dY, gW_);

			// gb = sum_rows(dY)
			Ops::SumRows(dY, gb_);

			// dX = dY * W^T  (전치 포함 커널)
			Ops::MatMulT_B(dY, W_, dX);
		}

		void ZeroGrad() override { gW_.Fill(0.0f); gb_.Fill(0.0f); }
		// Step는 Trainer에서 StudentUpdater로 처리하므로 no-op
		void Step(float) override {}


		// 접근자 (StudentUpdater용)
		Matrix& WRef() { return W_; }
		Matrix& bRef() { return b_; }
		Matrix& gWRef() { return gW_; }
		Matrix& gbRef() { return gb_; }
		const Matrix& W() const { return W_; }
		const Matrix& b() const { return b_; }
		const Matrix& gW() const { return gW_; }
		const Matrix& gb() const { return gb_; }
	};
}

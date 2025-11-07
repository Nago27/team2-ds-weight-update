// =============================
// include/vsnn/Dense.hpp (최종 최적화 3/4)
// (Ops.hpp의 병렬 함수를 호출하도록 연결)
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

		// Forward는 Ops::MatMul을 호출하므로 수정 불필요
		void Forward(const Matrix& X, Matrix& Y) override {
			Ops::MatMul(X, W_, Y);
			Ops::AddRowBias(Y, b_);
		}

		// [최적화] Backward의 3중 for 루프를
		// Ops.hpp의 병렬 함수 호출로 변경
		void Backward(const Matrix& X, const Matrix& dY, Matrix& dX) override {
			// gW = X^T * dY
			Ops::MatMul_AT(X, dY, gW_);

			// gb = sum_rows(dY)
			Ops::SumRows(dY, gb_);

			// dX = dY * W^T
			Ops::MatMul_BT(dY, W_, dX);
		}

		void ZeroGrad() override { gW_.Fill(0.0f); gb_.Fill(0.0f); }
		// Step는 Trainer에서 TrainUpdater로 처리하므로 no-op
		void Step(float) override {}


		// 접근자 (TrainUpdater용)
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


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

		// Forward: bias 융합 커널(행 접근) 사용 권장
		void Forward(const Matrix& X, Matrix& Y) override {
			Ops::MatMulBias(X, W_, b_, Y);  // 이 함수도 행-우선 누적 형태여야 함
		}

		void Backward(const Matrix& X, const Matrix& dY, Matrix& dX) override {
			// gW, gb 한 패스 누적을 쓰고 있었다면 그것 유지 OR 아래 2줄로 나눠도 됨
			// 1) gW 타일형 누적
			Ops::MatMulT_A_Tiled(X, dY, gW_);          // <-- REPLACE: 타일형으로 교체
			// 2) gb = sum_rows(dY) (또는 기존의 동시 누적 커널 유지)
			Ops::SumRows(dY, gb_);
			// 3) dX = dY * W^T (행 연속 접근 커널 그대로)
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

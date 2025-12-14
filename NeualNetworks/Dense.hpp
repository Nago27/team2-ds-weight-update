
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
		void Forward(const Matrix& X, Matrix& Y) override {
			Ops::MatMul1(X, W_, Y);
			Ops::AddRowBias(Y, b_);
		}
		void Backward(const Matrix& X, const Matrix& dY, Matrix& dX, int i) override {
			// gW = X^T * dY
			Ops::MatMul2(X, dY, gW_);

			// gb = sum_rows(dY)
			if (gb_.Rows() != 1 || gb_.Cols() != W_.Cols()) gb_.Reset(1, W_.Cols());
			float* gb_ptr = &gb_.Raw()[0];
			int num_cols = W_.Cols();

#pragma omp parallel 
			{
				vector<float> gb_local(W_.Cols(), 0.0f);
#pragma omp for
				for (i32 i = 0; i < X.Rows(); ++i) {
					const float* dY_ptr = &dY.Raw()[(size_t)i * num_cols];
					i32 j = 0;
					for (; j + 8 <= num_cols; j += 8) {
						__m256 dy = _mm256_loadu_ps(dY_ptr + j);
						__m256 gb = _mm256_loadu_ps(&gb_local[0] + j);
						gb = _mm256_add_ps(gb, dy);
						_mm256_storeu_ps(&gb_local[0] + j, gb);
					}
					for (; j < num_cols; ++j) {
						gb_local[j] += dY_ptr[j];
					}
				}
#pragma omp critical
				{
					for (i32 i = 0; i < W_.Cols(); ++i) {
						gb_ptr[i] += gb_local[i];
					}
				}
			}

			// dX = dY * W^T (첫번째 레이어에서는 생략)
			if (i != 0)
				Ops::MatMul3(dY, W_, dX);
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

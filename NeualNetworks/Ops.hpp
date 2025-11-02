// =============================
// include/vsnn/Ops.hpp
// =============================
#pragma once
#include <cmath>
#include <algorithm>
#include "Matrix.hpp"

using namespace std;

namespace vsnn {
	class Ops {
	public:
		// Y = X * W with shapes: (N,in) * (in,out) = (N,out)
		static void MatMul(const Matrix& A, const Matrix& B, Matrix& C) {
			int M = A.Rows(), K = A.Cols(), N = B.Cols();
			//캐시 효율성 최적화를 위해 B 행렬 전치
			const Matrix& Bt = B.Transposed();
			C.Reset(M, N);
		
			//캐시 블로킹을 통한 데이터 재사용률 극대화
			for (int i0 = 0; i0 < M; i0 += BLOCK) {
				int iMax = std::min(i0 + BLOCK, M);
				for (int j0 = 0; j0 < N; j0 += BLOCK) {
					int jMax = std::min(j0 + BLOCK, N);
					for (int k0 = 0; k0 < K; k0 += BLOCK) {
						int kMax = std::min(k0 + BLOCK, K);
						for (int i = i0; i < iMax; ++i) {
							const float* a = &A.Raw()[(size_t)i * K];
							for (int j = j0; j < jMax; ++j) {
								const float* b = &Bt.Raw()[(size_t)j * Bt.Cols()];
								float* c = &C.Raw()[(size_t)i * N];
								for (int k = k0; k < kMax; ++k)
									c[j] += a[k] * b[k];
							}
						}
					}
				}
			}
		}

		// 행렬곱 연산에서 뒤의 행렬이 전치행렬인 경우 굳이 전치행렬을 계산하지 않고 기존 행렬로 계산하는 것이 더 빠름
		static void MatMul_NT(const Matrix& A, const Matrix& B, Matrix& C) {
			int M = A.Rows(), K = A.Cols(), N = B.Rows(); //B가 아닌 B^T를 곱할 것이므로 N=B.Rows();
			// B^T를 곱할 것이므로 행렬 전치 필요 X
			C.Reset(M, N);
		
			for (int i0 = 0; i0 < M; i0 += BLOCK) {
				int iMax = std::min(i0 + BLOCK, M);
				for (int j0 = 0; j0 < N; j0 += BLOCK) {
					int jMax = std::min(j0 + BLOCK, N);
					for (int k0 = 0; k0 < K; k0 += BLOCK) {
						int kMax = std::min(k0 + BLOCK, K);
						for (int i = i0; i < iMax; ++i) {
							const float* a = &A.Raw()[(size_t)i * K];
							for (int j = j0; j < jMax; ++j) {
								const float* b = &B.Raw()[(size_t)j * B.Cols()];
								float* c = &C.Raw()[(size_t)i * N];
								for (int k = k0; k < kMax; ++k)
									c[j] += a[k] * b[k];
							}
						}
					}
				}
			}
		}

		static void AddRowBias(Matrix& Y, const Matrix& b) {
			assert(b.Rows() == 1 && b.Cols() == Y.Cols());
			int num_rows = Y.Rows();
			int num_cols = Y.Cols();
			const float* b_ptr = &b.Raw()[0];
			float* y_ptr = &Y.Raw()[0];

			for (int n = 0; n < num_rows; ++n) {
				float* y_ptr_ = y_ptr + num_cols * n;
				for (int j = 0; j < num_cols; ++j) {
					y_ptr_[j] += b_ptr[j];
				}
			}
		}

		static void ReLUForward(const Matrix& X, Matrix& Y) {
			if (Y.Rows() != X.Rows() || Y.Cols() != X.Cols()) Y.Reset(X.Rows(), X.Cols());

			const float* x_ptr = &X.Raw()[0];
			float* y_ptr = &Y.Raw()[0];
			int total_size = X.Rows() * X.Cols();
			
			for (int i = 0; i < total_size; ++i) {
				y_ptr[i] = (x_ptr[i] > 0.0f) ? x_ptr[i] : 0.0f;
			}
		}
		static void ReLUBackward(const Matrix& X, const Matrix& dY, Matrix& dX) {
			if (dX.Rows() != X.Rows() || dX.Cols() != X.Cols()) dX.Reset(X.Rows(), X.Cols());

			const float* x_ptr = &X.Raw()[0];
			const float* dy_ptr = &dY.Raw()[0];
			float* dx_ptr = &dX.Raw()[0];
			int total_size = X.Rows() * X.Cols();

			for (int i = 0; i < total_size; ++i) {
				dx_ptr[i] = (x_ptr[i] > 0.0f) ? dy_ptr[i] : 0.0f;
			}
		}
		static void SoftmaxRow(const float* logits, float* probs, int C) {
			float m = logits[0];
			for (int i = 1; i < C; ++i) m = max(m, logits[i]);
			float s = 0.0f; for (int i = 0; i < C; ++i) { probs[i] = exp(logits[i] - m); s += probs[i]; }
			if (s == 0.0f) s = 1e-12f;
			for (int i = 0; i < C; ++i) probs[i] /= s;
		}
	};
}

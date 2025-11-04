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
		// Y = X * W
		static void MatMul1(const Matrix& A, const Matrix& B, Matrix& C) {
			int M = A.Rows(), K = A.Cols(), N = B.Cols();
			if (C.Rows() != M || C.Cols() != N) C.Reset(M, N);

			for (int i = 0; i < M; ++i) {
				const float* a = &A.Raw()[(size_t)i * K];
				float* c = &C.Raw()[(size_t)i * N];
				for (int j = 0; j < K; ++j) {
					if (a[j] == 0.0f) continue;
					const float* b = &B.Raw()[(size_t)j * N];
					for (int k = 0; k < N; ++k) {
						c[k] += a[j] * b[k];
					}
				}
			}
		}
		// gW = X^T * dY
		static void MatMul2(const Matrix& A, const Matrix& B, Matrix& C) {
			int M = A.Rows(), K = A.Cols(), N = B.Cols();
			if (C.Rows() != K || C.Cols() != N) C.Reset(K, N);

			for (int i = 0; i < M; ++i) {
				const float* a = &A.Raw()[(size_t)i * K];
				const float* b = &B.Raw()[(size_t)i * N];
				for (int j = 0; j < K; ++j) {
					if (a[j] == 0.0f) continue;
					float* c = &C.Raw()[(size_t)j * N];
					for (int k = 0; k < N; ++k) {
						c[k] += a[j] * b[k];
					}
				}
			}

		}
		// dX = dY * W^T
		static void MatMul3(const Matrix& A, const Matrix& B, Matrix& C) {
			int M = A.Rows(), K = A.Cols(), N = B.Rows();
			if (C.Rows() != M || C.Cols() != N) C.Reset(M, N);

			Matrix BT(K, N);
			for (i32 i = 0; i < N; ++i) {
				const f32* src = &B.Raw()[(size_t)i * K];
				for (i32 j = 0; j < K; ++j) BT(j, i) = src[j];
			}

			for (int i = 0; i < M; ++i) {
				const float* a = &A.Raw()[(size_t)i * K];
				float* c = &C.Raw()[(size_t)i * N];
				for (int j = 0; j < K; ++j) {
					if (a[j] == 0) continue;
					const float* b = &BT.Raw()[(size_t)j * N];
					for (int k = 0; k < N; ++k) {
						c[k] += a[j] * b[k];
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

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
		static void MatMul(const Matrix& X, const Matrix& W, Matrix& Y) {
			assert(X.Cols() == W.Rows());
			if (Y.Rows() != X.Rows() || Y.Cols() != W.Cols()) Y.Reset(X.Rows(), W.Cols());
			for (i32 n = 0; n < X.Rows(); ++n) {
				for (i32 j = 0; j < W.Cols(); ++j) {
					float acc = 0.0f;
					for (i32 k = 0; k < X.Cols(); ++k) acc += X(n, k) * W(k, j);
					Y(n, j) = acc;
				}
			}
		}
		static void AddRowBias(Matrix& Y, const Matrix& b) {
			assert(b.Rows() == 1 && b.Cols() == Y.Cols());
			for (i32 n = 0; n < Y.Rows(); ++n)
				for (i32 j = 0; j < Y.Cols(); ++j) Y(n, j) += b(0, j);
		}
		static void ReLUForward(const Matrix& X, Matrix& Y) {
			if (Y.Rows() != X.Rows() || Y.Cols() != X.Cols()) Y.Reset(X.Rows(), X.Cols());
			for (i32 r = 0; r < X.Rows(); ++r)
				for (i32 c = 0; c < X.Cols(); ++c)
					Y(r, c) = (X(r, c) > 0.0f) ? X(r, c) : 0.0f;
		}
		static void ReLUBackward(const Matrix& X, const Matrix& dY, Matrix& dX) {
			if (dX.Rows() != X.Rows() || dX.Cols() != X.Cols()) dX.Reset(X.Rows(), X.Cols());
			for (i32 r = 0; r < X.Rows(); ++r)
				for (i32 c = 0; c < X.Cols(); ++c)
					dX(r, c) = (X(r, c) > 0.0f) ? dY(r, c) : 0.0f;
		}
		static void SoftmaxRow(const float* logits, float* probs, int C) {
			float m = logits[0];
			for (int i = 1; i < C; ++i) m = max(m, logits[i]);
			float s = 0.0f; for (int i = 0; i < C; ++i) { probs[i] = exp(logits[i] - m); s += probs[i]; }
			if (s == 0.0f) s = 1e-12f;
			for (int i = 0; i < C; ++i) probs[i] /= s;
		}

		// 벡터 방정식 및 전치 포함 연산 지원
		// ---- Forward: Y = X * W + b (루프 융합) --------------------------
		static void MatMulBias(const Matrix& X, const Matrix& W, const Matrix& b, Matrix& Y) {
			assert(X.Cols() == W.Rows());
			assert(b.Rows() == 1 && b.Cols() == W.Cols());
			const i32 N = X.Rows(), in = X.Cols(), out = W.Cols();
			if (Y.Rows() != N || Y.Cols() != out) Y.ResetNoInit(N, out);

			for (i32 n = 0; n < N; ++n) {
				const float* x = X.RowPtr(n);
				float* y = Y.RowPtr(n);
				for (i32 j = 0; j < out; ++j) {
					float acc = b(0, j);
					// W(k, j)는 row-major에서 행마다 연속
					for (i32 k = 0; k < in; ++k) acc += x[k] * W(k, j);
					y[j] = acc;
				}
			}
		}
		// ---- Backward: dX = dY * W^T ------------------------------------
		static void MatMulT_B(const Matrix& dY, const Matrix& W, Matrix& dX) {
			assert(dY.Cols() == W.Cols());
			const i32 N = dY.Rows(), out = dY.Cols(), in = W.Rows();
			if (dX.Rows() != N || dX.Cols() != in) dX.ResetNoInit(N, in);

			for (i32 n = 0; n < N; ++n) {
				const float* dy = dY.RowPtr(n);     // 연속
				float* dx = dX.RowPtr(n);
				for (i32 k = 0; k < in; ++k) {
					const float* wrow = W.RowPtr(k); // 연속
					float acc = 0.0f;
					for (i32 j = 0; j < out; ++j) acc += dy[j] * wrow[j];
					dx[k] = acc;
				}
			}
		}

		// ---- Backward: gW = X^T * dY  (outer-product 누적) --------------
		static void MatMulT_A(const Matrix& X, const Matrix& dY, Matrix& gW) {
			assert(X.Rows() == dY.Rows());
			const i32 N = X.Rows(), in = X.Cols(), out = dY.Cols();
			if (gW.Rows() != in || gW.Cols() != out) gW.ResetNoInit(in, out);
			gW.Fill(0.0f);

			for (i32 n = 0; n < N; ++n) {
				const float* x = X.RowPtr(n);
				const float* dy = dY.RowPtr(n);
				for (i32 k = 0; k < in; ++k) {
					float* gwrow = gW.RowPtr(k);   // 연속
					const float xk = x[k];
					for (i32 j = 0; j < out; ++j) gwrow[j] += xk * dy[j];
				}
			}
		}

		// ---- 행 합: gb = sum_rows(dY)  (편의) ---------------------------
		static void SumRows(const Matrix& A, Matrix& out1xC) {
			const i32 N = A.Rows(), C = A.Cols();
			if (out1xC.Rows() != 1 || out1xC.Cols() != C) out1xC.ResetNoInit(1, C);
			for (i32 j = 0; j < C; ++j) {
				float acc = 0.0f;
				for (i32 n = 0; n < N; ++n) acc += A(n, j);
				out1xC(0, j) = acc;
			}
		}
	};
}

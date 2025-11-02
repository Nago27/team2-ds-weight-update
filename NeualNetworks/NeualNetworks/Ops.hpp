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
		// ---- REPLACE: MatMulBias (행 연속 누적 + bias 사전 채움) ----
		static void MatMulBias(const Matrix& X, const Matrix& W, const Matrix& b, Matrix& Y) {
			assert(X.Cols() == W.Rows()); assert(b.Rows() == 1 && b.Cols() == W.Cols());
			const i32 N = X.Rows(), in = X.Cols(), out = W.Cols();
			if (Y.Rows() != N || Y.Cols() != out) Y.ResetNoInit(N, out);

			for (i32 n = 0; n < N; ++n) {
				const float* x = X.RowPtr(n);
				float* y = Y.RowPtr(n);

				// bias로 초기화
				for (i32 j = 0; j < out; ++j) y[j] = b(0, j);

				// y += x[k] * W.row(k)
				for (i32 k = 0; k < in; ++k) {
					const float s = x[k];
					if (s == 0.0f) continue;                 // 희소성 활용
					const float* wrow = W.RowPtr(k);       // 연속
					for (i32 j = 0; j < out; ++j) y[j] += s * wrow[j];
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

		// ReLU까지 1패스로 융합
		// ---- REPLACE: MatMulBiasReLU 본문 전체 교체 ----
		static void MatMulBiasReLU(const Matrix& X, const Matrix& W, const Matrix& b, Matrix& Y) {
			assert(X.Cols() == W.Rows()); assert(b.Rows() == 1 && b.Cols() == W.Cols());
			const i32 N = X.Rows(), in = X.Cols(), out = W.Cols();
			if (Y.Rows() != N || Y.Cols() != out) Y.ResetNoInit(N, out);

			for (i32 n = 0; n < N; ++n) {
				const float* x = X.RowPtr(n);
				float* y = Y.RowPtr(n);

				// 1) bias로 초기화 (행 단위 복사 가능)
				//   std::memcpy(y, b.RowPtr(0), sizeof(float)*out);  // memcpy 사용 가능
				for (i32 j = 0; j < out; ++j) y[j] = b(0, j);

				// 2) 행(연속) 기준 누적: y += x[k] * W.row(k)
				for (i32 k = 0; k < in; ++k) {
					const float s = x[k];
					if (s == 0.0f) continue;                  // 희소 입력이면 건너뛰기
					const float* wrow = W.RowPtr(k);          // 연속 접근
					for (i32 j = 0; j < out; ++j) y[j] += s * wrow[j];
				}

				// 3) ReLU 한 번에 적용
				for (i32 j = 0; j < out; ++j) y[j] = (y[j] > 0.f) ? y[j] : 0.f;
			}
		}

		// gb(bias grad)까지 한 번에 누적 (메모리 패스 삭제)
		static void MatMulT_A_GB(const Matrix& X, const Matrix& dY, Matrix& gW, Matrix& gb) {
			assert(X.Rows() == dY.Rows());
			const i32 N = X.Rows(), in = X.Cols(), out = dY.Cols();
			if (gW.Rows() != in || gW.Cols() != out) gW.ResetNoInit(in, out);
			if (gb.Rows() != 1 || gb.Cols() != out) gb.ResetNoInit(1, out);
			// gW.Fill(0.f);
			gb.Fill(0.f);

			for (i32 n = 0; n < N; ++n) {
				const float* x = X.RowPtr(n);
				const float* dy = dY.RowPtr(n);
				for (i32 k = 0; k < in; ++k) {
					float* gwrow = gW.RowPtr(k);
					const float xk = x[k];
					for (i32 j = 0; j < out; ++j) gwrow[j] += xk * dy[j];
				}
				// gb 누적 (같은 패스에서)
				for (i32 j = 0; j < out; ++j) gb(0, j) += dy[j];
			}
		}

		// ---- ADD: 타일형 gW 커널 (행 연속 + 캐시 블로킹) ----
		static void MatMulT_A_Tiled(const Matrix& X, const Matrix& dY, Matrix& gW,
			int Tk = 64, int Tj = 128) {
			assert(X.Rows() == dY.Rows());
			const i32 N = X.Rows(), in = X.Cols(), out = dY.Cols();
			if (gW.Rows() != in || gW.Cols() != out) gW.ResetNoInit(in, out);
			gW.Fill(0.0f);

			for (i32 kb = 0; kb < in; kb += Tk) {
				const i32 kend = std::min<i32>(in, kb + Tk);
				for (i32 jb = 0; jb < out; jb += Tj) {
					const i32 jend = std::min<i32>(out, jb + Tj);
					// 타일 누적
					for (i32 n = 0; n < N; ++n) {
						const float* x = X.RowPtr(n);
						const float* dy = dY.RowPtr(n);
						for (i32 k = kb; k < kend; ++k) {
							float* __restrict gwrow = gW.RowPtr(k) + jb;     // 연속
							const float s = x[k];
							if (s == 0.0f) continue; // ReLU 뒤 희소성 활용
							const float* __restrict dyblk = dy + jb;         // 연속
							for (i32 j = jb; j < jend; ++j) {
								gwrow[j - jb] += s * dyblk[j - jb];
							}
						}
					}
				}
			}
		}
	};
}

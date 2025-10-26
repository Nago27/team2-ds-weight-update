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

			const i32 N = X.Rows();
			const i32 K = X.Cols();
			const i32 J = W.Cols();
			if (Y.Rows() != N || Y.Cols() != J) Y.Reset(N, J);
			auto* x = const_cast<f32*>(X.Data());
			auto* w = const_cast<f32*>(W.Data());
			auto* y = Y.Data();

			// simple 2D blocking tuned for L1/L2 reuse (adjust if needed)
			const i32 BN = 64, BK = 64, BJ = 64;
			for (i32 nb = 0; nb < N; nb += BN) {
				const i32 nmax = std::min<i32>(N, nb + BN);
				for (i32 jb = 0; jb < J; jb += BJ) {
					const i32 jmax = std::min<i32>(J, jb + BJ);
					for (i32 kb = 0; kb < K; kb += BK) {
						const i32 kmax = std::min<i32>(K, kb + BK);
						for (i32 n = nb; n < nmax; ++n) {
							f32* yrow = y + (size_t)n * J;
							const f32* xrow = x + (size_t)n * K;
							for (i32 j = jb; j < jmax; ++j) {
								f32 acc = (kb == 0) ? 0.0f : yrow[j];
								for (i32 k = kb; k < kmax; ++k) {
									acc += xrow[k] * w[(size_t)k * J + j];
								}
								yrow[j] = acc;
							}
						}
					}
				}
			}


			/*
			if (Y.Rows() != X.Rows() || Y.Cols() != W.Cols()) Y.Reset(X.Rows(), W.Cols());
			for (i32 n = 0; n < X.Rows(); ++n) {
				for (i32 j = 0; j < W.Cols(); ++j) {
					float acc = 0.0f;
					for (i32 k = 0; k < X.Cols(); ++k) acc += X(n, k) * W(k, j);
					Y(n, j) = acc;
				}
			}*/
		}
		static void AddRowBias(Matrix& Y, const Matrix& b) {
			assert(b.Rows() == 1 && b.Cols() == Y.Cols());
			for (i32 n = 0; n < Y.Rows(); ++n) {
				f32* yrow = &Y.Raw()[(size_t)n * Y.Cols()];
				for (i32 j = 0; j < Y.Cols(); ++j) yrow[j] += b(0, j);
			}
		}

		// ReLU forward with mask (active indices per row)
		static void ReLUForwardWithMask(const Matrix& X, Matrix& Y, std::vector<int>& mask_index_flat) {
			const i32 R = X.Rows(), C = X.Cols();
			if (Y.Rows() != R || Y.Cols() != C) Y.Reset(R, C);
			mask_index_flat.clear();
			mask_index_flat.reserve((size_t)R * C / 2); // heuristic
			for (i32 r = 0; r < R; ++r) {
				for (i32 c = 0; c < C; ++c) {
					const f32 v = X(r, c);
					if (v > 0.0f) { Y(r, c) = v; mask_index_flat.push_back(r * C + c); }
					else { Y(r, c) = 0.0f; }
				}
			}
		}

		// Backward using mask (only write where active)
		static void ReLUBackwardWithMask(const std::vector<int>& mask_index_flat, const Matrix& dY, Matrix& dX) {
			if (dX.Rows() != dY.Rows() || dX.Cols() != dY.Cols()) dX.Reset(dY.Rows(), dY.Cols());
			// zero-out once per batch
			std::fill(dX.Raw().begin(), dX.Raw().end(), 0.0f);
			for (int idx : mask_index_flat) dX.Raw()[(size_t)idx] = dY.Raw()[(size_t)idx];
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
	};
}

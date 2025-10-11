// =============================
// include/vsnn/Ops.hpp
// =============================
#pragma once
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <thread>
#include "Matrix.hpp"

using namespace std;

namespace vsnn {
	//블록 사이즈 설정 (CPU마다 최적값이 다르므로 32, 64, 128 등 2^n으로 변경해가며 실행해봐야 함)
	const int BLOCK = 128;

	class Ops {
	public:
		static inline float Dot(const float* a, const float* b, int len) {

#ifdef __AVX2__
			int k = 0;
			__m256 acc = _mm256_setzero_ps();
			for (; k + 8 <= len; k += 8) {
				__m256 va = _mm256_loadu_ps(a + k);
				__m256 vb = _mm256_loadu_ps(b + k);
				acc = _mm256_fmadd_ps(va, vb, acc);
			}
			float tmp[8];
			_mm256_storeu_ps(tmp, acc);
			float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
			for (; k < len; ++k) s += a[k] * b[k];
			return s;
#else
			float s = 0.0f;
			for (int k = 0; k < len; ++k) s += a[k] * b[k];
			return s;
#endif
		}
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
							const float* a = &A.Raw()[(size_t)i * K + k0];
							for (int j = j0; j < jMax; ++j) {
								const float* b = &Bt.Raw()[(size_t)j * Bt.Cols() + k0];
								C(i, j) += Dot(a, b, kMax - k0);
							}
						}
					}
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
	};
}

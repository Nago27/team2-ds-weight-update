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
	class Ops {
	public:
		//AVX2 명령어를 지원하는 경우 SIMD를 이용해 8개의 요소를 동시에 처리 지원하지 않을경우 기본적인 스칼라 연산
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
							const float* a = &A.Raw()[(size_t)i * K + k0];
							for (int j = j0; j < jMax; ++j) {
								const float* b = &B.Raw()[(size_t)j * B.Cols() + k0];
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

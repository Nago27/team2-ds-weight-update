// =============================
// include/vsnn/Ops.hpp
// =============================
#pragma once
#include <cmath>
#include <algorithm> // std::min, std::max를 사용하기 위해 추가
#include "Matrix.hpp"  // 초기 std::vector 기반 Matrix
#include <stdexcept> // std::runtime_error를 사용하기 위해 추가

using namespace std;

namespace vsnn {
	class Ops {
	public:
		// Y = X * W with shapes: (N,K) * (K,M) = (N,M)
		// (원본 변수명 N, K, M을 사용)
		static void MatMul(const Matrix& X, const Matrix& W, Matrix& Y) {
			assert(X.Cols() == W.Rows());
			if (Y.Rows() != X.Rows() || Y.Cols() != W.Cols()) {
				Y.Reset(X.Rows(), W.Cols()); // Reset은 0으로 채우는 것을 가정
			}
			else {
				// 만약 Reset이 호출되지 않았다면, 
				// Y에 이전 값이 남아있을 수 있으므로 0으로 초기화합니다.
				// (초기 Matrix.hpp의 Fill 함수 사용)
				Y.Fill(0.0f);
			}

			const i32 N = X.Rows();
			const i32 K = X.Cols(); // ( = W.Rows() )
			const i32 M = W.Cols();

			// --- 캐시 블로킹 (Cache Blocking) 시작 ---

			// 1. 블록 크기 정의. 
			//    CPU L1/L2 캐시 크기에 맞춰야 합니다. (32, 64 등이 일반적)
			const int BLOCK_SIZE = 32;

			// 2. 바깥쪽 3중 루프: 블록(타일) 단위로 건너뜁니다.
			//    루프 순서는 캐시 효율성이 가장 좋은 (i, k, j) 순서를 따릅니다.
			for (i32 ii = 0; ii < N; ii += BLOCK_SIZE) { // N (행)
				for (i32 kk = 0; kk < K; kk += BLOCK_SIZE) { // K (공통 차원)
					for (i32 jj = 0; jj < M; jj += BLOCK_SIZE) { // M (열)

						// 3. 현재 블록의 실제 경계 계산
						//    (행렬 크기가 BLOCK_SIZE로 나눠 떨어지지 않을 경우 대비)
						const i32 i_end = std::min(ii + BLOCK_SIZE, N);
						const i32 k_end = std::min(kk + BLOCK_SIZE, K);
						const i32 j_end = std::min(jj + BLOCK_SIZE, M);

						// 4. 안쪽 3중 루프: 캐시에 로드된 작은 블록 내부에서만 계산
						for (i32 i = ii; i < i_end; ++i) {
							for (i32 k = kk; k < k_end; ++k) {
								// X(i, k) 값은 안쪽 루프(j) 내내 동일하므로
								// 미리 변수에 저장(캐싱)해 둡니다.
								const f32 x_ik = X(i, k);
								for (i32 j = jj; j < j_end; ++j) {
									// X(i, k)는 상수
									// W(k, j)는 메모리 순차 접근 (빠름)
									// Y(i, j)는 메모리 순차 접근 (빠름)
									Y(i, j) += x_ik * W(k, j);
								}
							}
						}
					}
				}
			}
			// --- 캐시 블로킹 끝 ---
		}

		// --- 나머지 함수들은 변경할 필요가 없습니다 ---

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
// =============================
// include/vsnn/Ops.hpp
// =============================
#pragma once
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include "Matrix.hpp"

using namespace std;

namespace vsnn {
    
	class Ops {
	public:
		// Y = X * W
		static void MatMul1(const Matrix& A, const Matrix& B, Matrix& C) {
			int M = A.Rows(), K = A.Cols(), N = B.Cols();
			C.Reset(M, N);
			if(K<N){
#pragma omp parallel for
				for (int i = 0; i < M; ++i) {
					const float* a = &A.Raw()[(size_t)i * K];
					float* c = &C.Raw()[(size_t)i * N];
					for (int j = 0; j < K; ++j) {
						if (a[j] == 0.0f) continue;
						const __m256 a_vec = _mm256_set1_ps(a[j]);
						const float* b = &B.Raw()[(size_t)j * N];
						int k = 0;
						for (; k + 8 < N; k += 8) {
							__m256 b_vec = _mm256_loadu_ps(b + k);
							__m256 c_vec = _mm256_loadu_ps(c + k);
							c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
							_mm256_storeu_ps(c + k, c_vec);
						}
						for (; k < N; k++) {
							c[k] += a[j] * b[k];
						}
					}
				}
			}
			else {
				Matrix BT(N, K);
#pragma omp parallel for
				for (i32 i = 0; i < K; ++i) {
					const f32* src = &B.Raw()[(size_t)i * N];
					for (i32 j = 0; j < N; ++j) BT(j, i) = src[j];
				}

#pragma omp parallel for
				for (int i = 0; i < M; ++i) {
					const float* a = &A.Raw()[(size_t)i * K];
					float* c = &C.Raw()[(size_t)i * N];
					for (int j = 0; j < N; ++j) {
						__m256 sum_vec = _mm256_setzero_ps();
						const float* b = &BT.Raw()[(size_t)j * K];
						int k = 0;
						for (; k + 8 < K; k += 8) {
							__m256 a_vec = _mm256_loadu_ps(a + k);
							__m256 b_vec = _mm256_loadu_ps(b + k);
							sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
						}
						float sum_array[8];
						_mm256_storeu_ps(sum_array, sum_vec);
						float final_sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
						for (; k < K; k++) {
							final_sum += a[k] * b[k];
						}
						c[j] = final_sum;
					}
				}
			}
		}
		// gW = X^T * dY
		static void MatMul2(const Matrix& A, const Matrix& B, Matrix& C) {
			int M = A.Rows(), K = A.Cols(), N = B.Cols();
			if (C.Rows() != K || C.Cols() != N) C.Reset(K, N);
			if (N > K) {
				Matrix AT(K, M);
#pragma omp parallel for
				for (i32 i = 0; i < M; ++i) {
					const f32* src = &A.Raw()[(size_t)i * K];
					for (i32 j = 0; j < K; ++j) AT(j, i) = src[j];
				}
#pragma omp parallel for
				for (int i = 0; i < K; ++i) {
					const float* a = &AT.Raw()[(size_t)i * M];
					float* c = &C.Raw()[(size_t)i * N];
					for (int j = 0; j < M; ++j) {
						if (a[j] == 0.0f) continue;
						const __m256 a_vec = _mm256_set1_ps(a[j]);					
						const float* b = &B.Raw()[(size_t)j * N];
						int k = 0;
						for (; k + 8 < N; k += 8) {
							__m256 b_vec = _mm256_loadu_ps(b + k);
							__m256 c_vec = _mm256_loadu_ps(c + k);
							c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
							_mm256_storeu_ps(c + k, c_vec);
						}
						for (; k < N; k++) {
							c[k] += a[j] * b[k];
						}

					}
				}
			}
			else {
				Matrix CT(N, K);
				for (int i = 0; i < M; ++i) {
					const float* a = &A.Raw()[(size_t)i * K];
					const float* b = &B.Raw()[(size_t)i * N];
#pragma omp parallel for
					for (int j = 0; j < N; ++j) {
						float* c = &CT.Raw()[(size_t)j * K];
						const __m256 b_vec = _mm256_set1_ps(b[j]);
						int k = 0;
						for (; k + 8 < K; k += 8) {
							const __m256 a_vec = _mm256_loadu_ps(a + k);
							__m256 c_vec = _mm256_loadu_ps(c + k);
							c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
							_mm256_storeu_ps(c + k, c_vec);
						}
						for (; k < K; k++) {
							c[k] += a[k] * b[j];
						}
					}
				}
#pragma omp parallel for
				for (i32 i = 0; i < N; ++i) {
					const f32* src = &CT.Raw()[(size_t)i * K];
					for (i32 j = 0; j < K; ++j) C(j, i) = src[j];
				}
			}

		}
		// dX = dY * W^T
		static void MatMul3(const Matrix& A, const Matrix& B, Matrix& C) {
			int M = A.Rows(), K = A.Cols(), N = B.Rows();
			C.Reset(M, N);

			Matrix BT(K, N);
#pragma omp parallel for
			for (i32 i = 0; i < N; ++i) {
				const f32* src = &B.Raw()[(size_t)i * K];
				for (i32 j = 0; j < K; ++j) BT(j, i) = src[j];
			}
#pragma omp parallel for
			for (int i = 0; i < M; ++i) {
				const float* a = &A.Raw()[(size_t)i * K];
				float* c = &C.Raw()[(size_t)i * N];
				for (int j = 0; j < K; ++j) {
					if (a[j] == 0.0f) continue;
					const __m256 a_vec = _mm256_set1_ps(a[j]);
					const float* b = &BT.Raw()[(size_t)j * N];
					int k = 0;
					for (; k + 8 < N; k += 8) {
						__m256 b_vec = _mm256_loadu_ps(b + k);
						__m256 c_vec = _mm256_loadu_ps(c + k);
						c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
						_mm256_storeu_ps(c + k, c_vec);
					}
					for (; k < N; k++) {
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
				int j = 0;
				// 8개씩 묶어 동시에 덧셈 연산
				for (; j + 8 <= num_cols; j += 8) {
					__m256 vy = _mm256_loadu_ps(y_ptr_ + j);
					__m256 vb = _mm256_loadu_ps(b_ptr + j);
					vy = _mm256_add_ps(vy, vb);
					_mm256_storeu_ps(y_ptr_ + j, vy);
				}

				// 남은 자투리 데이터 처리
				for (; j < num_cols; ++j) {
					y_ptr_[j] += b_ptr[j];
				}
			}
		}

		static void ReLUForward(const Matrix& X, Matrix& Y) {
			if (Y.Rows() != X.Rows() || Y.Cols() != X.Cols()) Y.Reset(X.Rows(), X.Cols());

			const float* x_ptr = &X.Raw()[0];
			float* y_ptr = &Y.Raw()[0];
			int total_size = X.Rows() * X.Cols();
			int i = 0;
			// 0으로 채워진 256비트 벡터를 미리 준비
			const __m256 zeros = _mm256_setzero_ps();
			// 8개씩 묶어 0과 비교
			for (; i + 8 <= total_size; i += 8) {
				__m256 vx = _mm256_loadu_ps(x_ptr + i);
				// 8개의 float 각각에 대해 0과 비교하여 큰 값을 선택
				__m256 vy = _mm256_max_ps(vx, zeros);
				_mm256_storeu_ps(y_ptr + i, vy);
			}
			// 남은 자투리 데이터 처리
			for (; i < total_size; ++i) {
				y_ptr[i] = (x_ptr[i] > 0.0f) ? x_ptr[i] : 0.0f;
			}
		}

		static void ReLUBackward(const Matrix& X, const Matrix& dY, Matrix& dX) {
			if (dX.Rows() != X.Rows() || dX.Cols() != X.Cols()) dX.Reset(X.Rows(), X.Cols());

			const float* x_ptr = &X.Raw()[0];
			const float* dy_ptr = &dY.Raw()[0];
			float* dx_ptr = &dX.Raw()[0];
			int total_size = X.Rows() * X.Cols();
			int i = 0;
			const __m256 zeros = _mm256_setzero_ps();
			for (; i + 8 <= total_size; i += 8) {
				__m256 vx = _mm256_loadu_ps(x_ptr + i);
				__m256 vdy = _mm256_loadu_ps(dy_ptr + i);
				// 1. 마스크 생성: vx의 8개 원소가 각각 0보다 큰지 비교 (크면 1, 아니면 0)
				__m256 mask = _mm256_cmp_ps(vx, zeros, _CMP_GT_OQ);

				// 2. 블렌드: 마스크를 이용해 vdy와 0을 섞음
				// 마스크 비트가 1인 위치는 vdy 값을, 0인 위치는 0 값을 선택
				__m256 vdx = _mm256_blendv_ps(zeros, vdy, mask);

				_mm256_storeu_ps(dx_ptr + i, vdx);
			}
			// 남은 자투리 데이터 처리
			for (; i < total_size; ++i) {
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

// =============================
// include/vsnn/Ops.hpp (최종 수정본: 작은 배치 딜레이 해결)
// (main.cpp의 테스트 루프 병목 해결)
// =============================
#pragma once
#include <cmath>
#include <algorithm>
#include <vector>
#include <thread>
#include <numeric>
#include <cassert>
#include "Matrix.hpp"

using namespace std;

namespace vsnn {

    class Ops {
    private:
        // 캐시 효율을 위한 타일(블록) 크기
        static constexpr i32 TILE_SIZE = 16;

        // [수정] 스레드를 생성할 최소한의 작업량(행 수)
        // 이 값보다 작으면 싱글스레드가 더 빠름
        static constexpr i32 PARALLEL_THRESHOLD = 64;

        // [버그 수정] num_threads가 0이 되어 0으로 나누기 오류가 발생하는 것을 방지
        static i32 GetNumThreads() {
            i32 n = static_cast<i32>(std::thread::hardware_concurrency());
            return (n == 0) ? 1 : n;
        }

        // --- 원본 싱글스레드 함수들 (작은 배치를 위해 복원) ---
        static void MatMul_Single(const Matrix& A, const Matrix& B, Matrix& C) {
            const i32 M = A.Rows(); const i32 K = A.Cols(); const i32 N = B.Cols();
            if (C.Rows() != M || C.Cols() != N) C.Reset(M, N);

            for (i32 i = 0; i < M; ++i) {
                for (i32 j = 0; j < N; ++j) {
                    float acc = 0.0f;
                    for (i32 k = 0; k < K; ++k) acc += A(i, k) * B(k, j);
                    C(i, j) = acc;
                }
            }
        }
        static void MatMul_AT_Single(const Matrix& A, const Matrix& B, Matrix& C) {
            const i32 K = A.Rows(); const i32 M = A.Cols(); const i32 N = B.Cols();
            if (C.Rows() != M || C.Cols() != N) C.Reset(M, N);

            for (i32 i = 0; i < M; ++i) { // C의 행 (A의 열)
                for (i32 j = 0; j < N; ++j) { // C의 열 (B의 열)
                    float acc = 0.0f;
                    for (i32 k = 0; k < K; ++k) acc += A(k, i) * B(k, j); // A(k,i) * B(k,j)
                    C(i, j) = acc;
                }
            }
        }
        static void MatMul_BT_Single(const Matrix& A, const Matrix& B, Matrix& C) {
            const i32 M = A.Rows(); const i32 K = A.Cols(); const i32 N = B.Rows();
            if (C.Rows() != M || C.Cols() != N) C.Reset(M, N);

            for (i32 i = 0; i < M; ++i) { // C의 행 (A의 행)
                for (i32 j = 0; j < N; ++j) { // C의 열 (B의 행)
                    float acc = 0.0f;
                    for (i32 k = 0; k < K; ++k) acc += A(i, k) * B(j, k); // A(i,k) * B(j,k)
                    C(i, j) = acc;
                }
            }
        }
        static void SumRows_Single(const Matrix& A, Matrix& Y) {
            const i32 N = A.Rows(); const i32 D = A.Cols();
            if (Y.Rows() != 1 || Y.Cols() != D) Y.Reset(1, D);
            Y.Fill(0.0f);

            for (i32 i = 0; i < N; ++i) {
                for (i32 j = 0; j < D; ++j) {
                    Y(0, j) += A(i, j);
                }
            }
        }
        // --- (복원 끝) ---


        // [최적화] 수동 스레딩을 사용하는 타일 기반 행렬 곱셈 (Y = A * B)
        // i-k-j 루프 순서로 캐시 효율 극대화
        static void MatMul_Tiled_Parallel(const Matrix& A, const Matrix& B, Matrix& C) {
            const i32 M = A.Rows(); const i32 K = A.Cols(); const i32 N = B.Cols();
            if (C.Rows() != M || C.Cols() != N) C.Reset(M, N);
            C.Fill(0.0f);

            // [자료구조] 스레드 객체를 관리하기 위한 std::vector
            const i32 num_threads = GetNumThreads();
            std::vector<std::thread> threads(num_threads);
            const i32 rows_per_thread = (M + num_threads - 1) / num_threads;

            for (i32 t = 0; t < num_threads; ++t) {
                const i32 row_start = t * rows_per_thread;
                const i32 row_end = std::min(row_start + rows_per_thread, M);
                if (row_start >= row_end) break; // 작업이 없으면 스레드 생성 안함

                // 스레드 생성 오버헤드가 MatMul 함수 당 1번만 발생
                threads[t] = std::thread([row_start, row_end, K, N, &A, &B, &C] {
                    const f32* a_data = A.Data();
                    const f32* b_data = B.Data();
                    f32* c_data = C.Data();

                    for (i32 ii = row_start; ii < row_end; ii += TILE_SIZE) {
                        for (i32 kk = 0; kk < K; kk += TILE_SIZE) {
                            for (i32 jj = 0; jj < N; jj += TILE_SIZE) {
                                // 타일(블록) 내부 계산
                                const i32 i_max = std::min(ii + TILE_SIZE, row_end);
                                for (i32 i = ii; i < i_max; ++i) {
                                    const i32 k_max = std::min(kk + TILE_SIZE, K);
                                    for (i32 k = kk; k < k_max; ++k) {
                                        const i32 j_max = std::min(jj + TILE_SIZE, N);
                                        const f32 A_ik = a_data[i * K + k]; // A(i, k)
                                        f32* C_i_ptr = c_data + i * N; // C(i, *) 포인터
                                        const f32* B_k_ptr = b_data + k * N; // B(k, *) 포인터

                                        for (i32 j = jj; j < j_max; ++j) {
                                            C_i_ptr[j] += A_ik * B_k_ptr[j]; // C(i, j) += A(i, k) * B(k, j)
                                        }
                                    }
                                }
                            }
                        }
                    }
                    });
            }
            // 모든 MatMul 작업이 끝날 때까지 대기
            for (i32 t = 0; t < num_threads; ++t) {
                if (threads[t].joinable()) threads[t].join();
            }
        }

        // [최적화] 수동 스레딩을 사용하는 타일 기반 행렬 곱셈 (Y = A^T * B)
        static void MatMul_AT_Tiled_Parallel(const Matrix& A, const Matrix& B, Matrix& C) {
            const i32 K = A.Rows(); const i32 M = A.Cols(); const i32 N = B.Cols();
            if (C.Rows() != M || C.Cols() != N) C.Reset(M, N);
            C.Fill(0.0f);

            const i32 num_threads = GetNumThreads();
            std::vector<std::thread> threads(num_threads);
            const i32 cols_per_thread = (M + num_threads - 1) / num_threads;

            for (i32 t = 0; t < num_threads; ++t) {
                const i32 col_start = t * cols_per_thread;
                const i32 col_end = std::min(col_start + cols_per_thread, M);
                if (col_start >= col_end) break;

                threads[t] = std::thread([col_start, col_end, M, K, N, &A, &B, &C] {
                    const f32* a_data = A.Data();
                    const f32* b_data = B.Data();
                    f32* c_data = C.Data();

                    for (i32 kk = 0; kk < K; kk += TILE_SIZE) {
                        for (i32 ii = col_start; ii < col_end; ii += TILE_SIZE) {
                            for (i32 jj = 0; jj < N; jj += TILE_SIZE) {
                                const i32 k_max = std::min(kk + TILE_SIZE, K);
                                for (i32 k = kk; k < k_max; ++k) {
                                    const i32 i_max = std::min(ii + TILE_SIZE, col_end);
                                    for (i32 i = ii; i < i_max; ++i) {
                                        const i32 j_max = std::min(jj + TILE_SIZE, N);
                                        const f32 A_ki = a_data[k * M + i]; // A(k, i)
                                        f32* C_i_ptr = c_data + i * N;
                                        const f32* B_k_ptr = b_data + k * N;

                                        for (i32 j = jj; j < j_max; ++j) {
                                            C_i_ptr[j] += A_ki * B_k_ptr[j];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    });
            }
            for (i32 t = 0; t < num_threads; ++t) {
                if (threads[t].joinable()) threads[t].join();
            }
        }

        // [최적화] 수동 스레딩을 사용하는 타일 기반 행렬 곱셈 (Y = A * B^T)
        static void MatMul_BT_Tiled_Parallel(const Matrix& A, const Matrix& B, Matrix& C) {
            const i32 M = A.Rows(); const i32 K = A.Cols(); const i32 N = B.Rows();
            if (C.Rows() != M || C.Cols() != N) C.Reset(M, N);
            C.Fill(0.0f);

            const i32 num_threads = GetNumThreads();
            std::vector<std::thread> threads(num_threads);
            const i32 rows_per_thread = (M + num_threads - 1) / num_threads;

            for (i32 t = 0; t < num_threads; ++t) {
                const i32 row_start = t * rows_per_thread;
                const i32 row_end = std::min(row_start + rows_per_thread, M);
                if (row_start >= row_end) break;

                threads[t] = std::thread([row_start, row_end, K, N, &A, &B, &C] {
                    const f32* a_data = A.Data();
                    const f32* b_data = B.Data();
                    f32* c_data = C.Data();

                    for (i32 ii = row_start; ii < row_end; ii += TILE_SIZE) {
                        for (i32 jj = 0; jj < N; jj += TILE_SIZE) {
                            for (i32 kk = 0; kk < K; kk += TILE_SIZE) {
                                const i32 i_max = std::min(ii + TILE_SIZE, row_end);
                                for (i32 i = ii; i < i_max; ++i) {
                                    const i32 j_max = std::min(jj + TILE_SIZE, N);
                                    for (i32 j = jj; j < j_max; ++j) {
                                        const i32 k_max = std::min(kk + TILE_SIZE, K);
                                        f32 sum = 0.0f;
                                        f32* C_ij_ptr = c_data + i * N + j;

                                        for (i32 k = kk; k < k_max; ++k) {
                                            sum += a_data[i * K + k] * b_data[j * K + k];
                                        }
                                        *C_ij_ptr += sum;
                                    }
                                }
                            }
                        }
                    }
                    });
            }
            for (i32 t = 0; t < num_threads; ++t) {
                if (threads[t].joinable()) threads[t].join();
            }
        }

        // [최적화] 행렬의 행들을 병렬로 합산 (Y = sum_rows(A))
        static void SumRows_Parallel(const Matrix& A, Matrix& Y) {
            const i32 N = A.Rows(); const i32 D = A.Cols();
            if (Y.Rows() != 1 || Y.Cols() != D) Y.Reset(1, D);
            Y.Fill(0.0f);

            const i32 num_threads = GetNumThreads();
            std::vector<std::thread> threads(num_threads);

            // [자료구조] 각 스레드가 자신의 로컬 합계(1 x D)를 계산할 임시 저장소
            std::vector<Matrix> partial_sums(num_threads);
            for (i32 t = 0; t < num_threads; ++t) {
                partial_sums[t].Reset(1, D);
            }

            const i32 rows_per_thread = (N + num_threads - 1) / num_threads;

            for (i32 t = 0; t < num_threads; ++t) {
                const i32 row_start = t * rows_per_thread;
                const i32 row_end = std::min(row_start + rows_per_thread, N);
                if (row_start >= row_end) break;

                threads[t] = std::thread([row_start, row_end, D, &A, &partial_sums, t] {
                    const f32* a_data = A.Data();
                    f32* p_sum_data = partial_sums[t].Data();
                    for (i32 i = row_start; i < row_end; ++i) {
                        const f32* A_i_ptr = a_data + i * D;
                        for (i32 j = 0; j < D; ++j) {
                            p_sum_data[j] += A_i_ptr[j];
                        }
                    }
                    });
            }
            for (i32 t = 0; t < num_threads; ++t) {
                if (threads[t].joinable()) threads[t].join();
            }

            // 모든 스레드의 부분 합계를 최종 Y에 더함
            f32* y_data = Y.Data();
            for (i32 t = 0; t < num_threads; ++t) {
                const f32* p_sum_data = partial_sums[t].Data();
                for (i32 j = 0; j < D; ++j) {
                    y_data[j] += p_sum_data[j];
                }
            }
        }


    public:
        // Y = X * W with shapes: (N,in) * (in,out) = (N,out)
        static void MatMul(const Matrix& X, const Matrix& W, Matrix& Y) {
            // [수정] 입력 행(M)의 크기에 따라 분기
            if (X.Rows() < PARALLEL_THRESHOLD) {
                MatMul_Single(X, W, Y); // (테스트 루프용)
            }
            else {
                MatMul_Tiled_Parallel(X, W, Y); // (훈련 루프용)
            }
        }
        // Y = X^T * W
        static void MatMul_AT(const Matrix& X, const Matrix& W, Matrix& Y) {
            // [수정] 입력 행(K)의 크기에 따라 분기
            if (X.Rows() < PARALLEL_THRESHOLD) {
                MatMul_AT_Single(X, W, Y);
            }
            else {
                MatMul_AT_Tiled_Parallel(X, W, Y);
            }
        }
        // Y = X * W^T
        static void MatMul_BT(const Matrix& X, const Matrix& W, Matrix& Y) {
            // [수정] 입력 행(M)의 크기에 따라 분기
            if (X.Rows() < PARALLEL_THRESHOLD) {
                MatMul_BT_Single(X, W, Y);
            }
            else {
                MatMul_BT_Tiled_Parallel(X, W, Y);
            }
        }
        // Y(1,D) = sum_rows(X(N,D))
        static void SumRows(const Matrix& X, Matrix& Y) {
            // [수정] 입력 행(N)의 크기에 따라 분기
            if (X.Rows() < PARALLEL_THRESHOLD) {
                SumRows_Single(X, Y);
            }
            else {
                SumRows_Parallel(X, Y);
            }
        }

        // [최적화] 가벼운 함수들은 스레드 생성 오버헤드를 피하기 위해
        // 다시 싱글스레드 루프로 롤백합니다. (이것이 더 빠름)
        static void AddRowBias(Matrix& Y, const Matrix& b) {
            assert(b.Rows() == 1 && b.Cols() == Y.Cols());
            const i32 N = Y.Rows(); const i32 D = Y.Cols();
            const f32* b_data = b.Data();
            f32* y_data = Y.Data();

            for (i32 n = 0; n < N; ++n) {
                f32* Y_n_ptr = y_data + n * D;
                for (i32 j = 0; j < D; ++j) {
                    Y_n_ptr[j] += b_data[j];
                }
            }
        }

        // [최적화] 싱글스레드 롤백 (가장 빠름)
        static void ReLUForward(const Matrix& X, Matrix& Y) {
            if (Y.Rows() != X.Rows() || Y.Cols() != X.Cols()) Y.Reset(X.Rows(), X.Cols());
            const size_t total_size = static_cast<size_t>(X.Rows()) * X.Cols();
            const f32* x_data = X.Data();
            f32* y_data = Y.Data();

            for (size_t i = 0; i < total_size; ++i) {
                y_data[i] = (x_data[i] > 0.0f) ? x_data[i] : 0.0f;
            }
        }

        // [최적화] 싱글스레드 롤백 (가장 빠름)
        static void ReLUBackward(const Matrix& X, const Matrix& dY, Matrix& dX) {
            if (dX.Rows() != X.Rows() || dX.Cols() != X.Cols()) dX.Reset(X.Rows(), X.Cols());
            const size_t total_size = static_cast<size_t>(X.Rows()) * X.Cols();
            const f32* x_data = X.Data();
            const f32* dy_data = dY.Data();
            f32* dx_data = dX.Data();

            for (size_t i = 0; i < total_size; ++i) {
                dx_data[i] = (x_data[i] > 0.0f) ? dy_data[i] : 0.0f;
            }
        }

        // Softmax는 배치 크기(N)만큼 루프를 돌지만, 각 연산(C)이 작으므로
        // 병렬화 오버헤드가 더 클 수 있습니다. 싱글스레드로 유지합니다.
        static void SoftmaxRow(const float* logits, float* probs, int C) {
            float m = logits[0];
            for (int i = 1; i < C; ++i) m = max(m, logits[i]);
            float s = 0.0f;
            for (int i = 0; i < C; ++i) {
                probs[i] = exp(logits[i] - m);
                s += probs[i];
            }
            if (s == 0.0f) s = 1e-12f; // 수치적 안정성
            const float inv_s = 1.0f / s;
            for (int i = 0; i < C; ++i) probs[i] *= inv_s;
        }
    };
}


// =============================
// include/vsnn/Ops.hpp
// =============================
#pragma once
#include <cmath>
#include <algorithm>
#include <thread>
#include<queue>
#include<mutex>
#include<condition_variable>
#include<functional>
#include "Matrix.hpp"

using namespace std;

namespace vsnn {
    class ThreadPool {
    public:
        // 생성자: 하드웨어 코어 수만큼 스레드를 미리 생성합니다.
        ThreadPool(size_t num_threads) : stop(false) {
            if (num_threads == 0) num_threads = 1;
            for (size_t i = 0; i < num_threads; ++i) {
                workers.emplace_back([this] {
                    // 각 스레드가 이 '일꾼 루프'를 계속 돕니다.
                    for (;;) {
                        std::function<void()> task;
                        {
                            // 1. 뮤텍스로 큐를 잠급니다.
                            std::unique_lock<std::mutex> lock(this->queue_mutex);

                            // 2. '일거리가 없으면' 잠듭니다. (stop 신호가 와도 깸)
                            this->condition.wait(lock, [this] {
                                return this->stop || !this->tasks.empty();
                                });

                            // 3. 종료 신호가 왔고, 일거리도 없으면 스레드 종료
                            if (this->stop && this->tasks.empty())
                                return;

                            // 4. 일거리를 하나 꺼냅니다.
                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                        } // 5. 큐의 잠금을 해제합니다. (다른 스레드가 큐에 접근 가능)

                        // 6. 일거리를 실행합니다.
                        task();
                    }
                 });
            }
        }

        ~ThreadPool() {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                stop = true; // 1. 종료 신호
            }
            condition.notify_all(); // 2. 자고 있는 모든 스레드를 깨움
            for (std::thread& worker : workers)
                worker.join(); // 3. 모든 스레드가 종료될 때까지 대기
        }

        // '작업표' (함수)를 큐에 넣는 함수
        void enqueue(std::function<void()> task) {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (stop) return; // 풀이 닫혔으면 작업을 받지 않음
                tasks.emplace(std::move(task));
            }
            condition.notify_one(); // 자고 있는 스레드 중 '하나'를 깨움
        }

    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
        std::mutex queue_mutex;
        std::condition_variable condition;
        bool stop;
    };

    ThreadPool g_pool(std::thread::hardware_concurrency());

	class Ops {
	public:
        
        // Y = X * W
        static void MatMul1(const Matrix& A, const Matrix& B, Matrix& C) {
            int M = A.Rows(), K = A.Cols(), N = B.Cols();
            if (C.Rows() != M || C.Cols() != N) C.Reset(M, N);

            unsigned int num_threads = std::thread::hardware_concurrency();
            if (M < num_threads) num_threads = M;

            int rows_per_thread = (M + num_threads - 1) / num_threads;
 
            std::atomic<int> tasks_remaining(num_threads);

            for (unsigned int t = 0; t < num_threads; ++t) {
                int start_row = t * rows_per_thread;
                int end_row = std::min(start_row + rows_per_thread, M);

                if (start_row >= end_row) {
                    tasks_remaining.fetch_sub(1); // 이 스레드는 할 일이 없음
                    continue;
                }

                auto task = [&A, &B, &C, start_row, end_row, K, N, &tasks_remaining]() {
                    for (int i = start_row; i < end_row; ++i) {
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
                    tasks_remaining.fetch_sub(1);
                 };
                g_pool.enqueue(std::move(task));
            }

            while (tasks_remaining.load() > 0) {
                std::this_thread::yield();
            }
        }

        // gW = X^T * dY
        static void MatMul2(const Matrix& A, const Matrix& B, Matrix& C) {
            int M = A.Rows(), K = A.Cols(), N = B.Cols();
            if (C.Rows() != K || C.Cols() != N) C.Reset(K, N);

            unsigned int num_threads = std::thread::hardware_concurrency();
            if (K < num_threads) num_threads = K;

            int rows_per_thread = (K + num_threads - 1) / num_threads;

            std::atomic<int> tasks_remaining(num_threads);

            for (unsigned int t = 0; t < num_threads; ++t) {
                int start_row = t * rows_per_thread;
                int end_row = std::min(start_row + rows_per_thread, K);

                if (start_row >= end_row) {
                    tasks_remaining.fetch_sub(1);
                    continue;
                }

                auto task = [&A, &B, &C, start_row, end_row, M, K, N, &tasks_remaining]() {
                    for (int i = 0; i < M; ++i) {
                        const float* a = &A.Raw()[(size_t)i * K];
                        const float* b = &B.Raw()[(size_t)i * N];
                        for (int j = start_row; j < end_row; ++j) {
                            if (a[j] == 0.0f) continue;
                            float* c = &C.Raw()[(size_t)j * N];
                            for (int k = 0; k < N; ++k) {
                                c[k] += a[j] * b[k];
                            }
                        }
                    }
                    tasks_remaining.fetch_sub(1);
                 };
                g_pool.enqueue(std::move(task));
            }
            while (tasks_remaining.load() > 0) {
                std::this_thread::yield();
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

            
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (M < num_threads) num_threads = M;

            int rows_per_thread = (M + num_threads - 1) / num_threads;

            std::atomic<int> tasks_remaining(num_threads);

            for (unsigned int t = 0; t < num_threads; ++t) {
                int start_row = t * rows_per_thread;
                int end_row = std::min(start_row + rows_per_thread, M);

                if (start_row >= end_row) {
                    tasks_remaining.fetch_sub(1); 
                    continue;
                }

                auto task = [&A, &BT, &C, start_row, end_row, K, N, &tasks_remaining]() {
                    for (int i = start_row; i < end_row; ++i) { 
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
                    tasks_remaining.fetch_sub(1);
                 };
                g_pool.enqueue(std::move(task));
            }
            while (tasks_remaining.load() > 0) {
                std::this_thread::yield();
            }
        }
        

		//// Y = X * W
		//static void MatMul1(const Matrix& A, const Matrix& B, Matrix& C) {
		//	int M = A.Rows(), K = A.Cols(), N = B.Cols();
		//	if (C.Rows() != M || C.Cols() != N) C.Reset(M, N);

		//	for (int i = 0; i < M; ++i) {
		//		const float* a = &A.Raw()[(size_t)i * K];
		//		float* c = &C.Raw()[(size_t)i * N];
		//		for (int j = 0; j < K; ++j) {
		//			if (a[j] == 0.0f) continue;
		//			const float* b = &B.Raw()[(size_t)j * N];
		//			for (int k = 0; k < N; ++k) {
		//				c[k] += a[j] * b[k];
		//			}
		//		}
		//	}
		//}
		//// gW = X^T * dY
		//static void MatMul2(const Matrix& A, const Matrix& B, Matrix& C) {
		//	int M = A.Rows(), K = A.Cols(), N = B.Cols();
		//	if (C.Rows() != K || C.Cols() != N) C.Reset(K, N);

		//	for (int i = 0; i < M; ++i) {
		//		const float* a = &A.Raw()[(size_t)i * K];
		//		const float* b = &B.Raw()[(size_t)i * N];
		//		for (int j = 0; j < K; ++j) {
		//			if (a[j] == 0.0f) continue;
		//			float* c = &C.Raw()[(size_t)j * N];
		//			for (int k = 0; k < N; ++k) {
		//				c[k] += a[j] * b[k];
		//			}
		//		}
		//	}

		//}
		//// dX = dY * W^T
		//static void MatMul3(const Matrix& A, const Matrix& B, Matrix& C) {
		//	int M = A.Rows(), K = A.Cols(), N = B.Rows();
		//	if (C.Rows() != M || C.Cols() != N) C.Reset(M, N);

		//	Matrix BT(K, N);
		//	for (i32 i = 0; i < N; ++i) {
		//		const f32* src = &B.Raw()[(size_t)i * K];
		//		for (i32 j = 0; j < K; ++j) BT(j, i) = src[j];
		//	}

		//	for (int i = 0; i < M; ++i) {
		//		const float* a = &A.Raw()[(size_t)i * K];
		//		float* c = &C.Raw()[(size_t)i * N];
		//		for (int j = 0; j < K; ++j) {
		//			if (a[j] == 0) continue;
		//			const float* b = &BT.Raw()[(size_t)j * N];
		//			for (int k = 0; k < N; ++k) {
		//				c[k] += a[j] * b[k];
		//			}
		//		}
		//	}
		//}

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

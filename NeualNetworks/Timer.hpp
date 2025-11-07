// =============================
// include/vsnn/Timer.hpp (최종 최적화 4/4)
// (TrainUpdater 병렬화)
// =============================

// --- Part 1: Timer Class (Unchanged) ---
#pragma once
#include <chrono>

using namespace std;

namespace vsnn {
    class Timer {
    public:
        using clock = chrono::high_resolution_clock;
    private:
        clock::time_point t0_;
    public:
        void Tic() { t0_ = clock::now(); }
        double TocMs() const {
            auto t1 = clock::now();
            return chrono::duration<double, milli>(t1 - t0_).count();
        }
    };
}


// --- Part 2: TrainUpdater Class (Optimized) ---
#pragma once
#include <type_traits>
#include <vector>       // [최적화] std::vector
#include <thread>       // [최적화] std::thread
#include <numeric>      // for std::min
#include "Sequential.hpp"
#include "Dense.hpp"
// Timer.hpp (자기 자신)를 다시 include할 필요가 없으므로 원본의 include 삭제

namespace vsnn {
    class TrainUpdater {
    private:
        // [버그 수정] num_threads가 0이 되어 0으로 나누기 오류가 발생하는 것을 방지
        static i32 GetNumThreads() {
            i32 n = static_cast<i32>(std::thread::hardware_concurrency());
            return (n == 0) ? 1 : n;
        }

    public:
        // [최적화] W <- W - lr * gW, b <- b - lr * gb 를 병렬로 수행
        static void Update(Sequential& model, float lr) {
            for (size_t li = 0; li < model.NumLayers(); ++li) {
                auto* L = model.LayerAt(li);
                auto* D = dynamic_cast<Dense*>(L);
                if (!D) continue;

                Matrix& W = D->WRef(); Matrix& gW = D->gWRef();
                Matrix& b = D->bRef(); Matrix& gb = D->gbRef();

                const i32 num_threads = GetNumThreads();
                // [자료구조] 스레드 객체 관리를 위한 std::vector
                std::vector<std::thread> threads(num_threads);

                // 1. W (가중치) 병렬 업데이트
                // (W는 크기가 크므로 행(rows) 기준으로 작업을 분배)
                const i32 W_rows = W.Rows();
                const i32 rows_per_thread = (W_rows + num_threads - 1) / num_threads;

                for (i32 t = 0; t < num_threads; ++t) {
                    const i32 row_start = t * rows_per_thread;
                    const i32 row_end = std::min(row_start + rows_per_thread, W_rows);

                    threads[t] = std::thread([row_start, row_end, &W, &gW, lr] {
                        const i32 num_cols = W.Cols();
                        f32* w_data = W.Data();
                        const f32* gw_data = gW.Data();

                        // 1차원 배열이므로 시작 인덱스 계산
                        size_t start_idx = static_cast<size_t>(row_start) * num_cols;
                        size_t end_idx = static_cast<size_t>(row_end) * num_cols;

                        for (size_t i = start_idx; i < end_idx; ++i) {
                            w_data[i] -= lr * gw_data[i];
                        }
                        });
                }
                for (auto& th : threads) th.join();

                // 2. b (편향) 병렬 업데이트
                // (b는 1 x D 벡터이므로 열(cols) 기준으로 작업을 분배)
                const i32 b_cols = b.Cols();
                const i32 cols_per_thread = (b_cols + num_threads - 1) / num_threads;

                // 스레드 벡터 재사용
                threads.clear();
                threads.resize(num_threads);

                for (i32 t = 0; t < num_threads; ++t) {
                    const i32 col_start = t * cols_per_thread;
                    const i32 col_end = std::min(col_start + cols_per_thread, b_cols);

                    threads[t] = std::thread([col_start, col_end, &b, &gb, lr] {
                        f32* b_data = b.Data();
                        const f32* gb_data = gb.Data();
                        for (i32 j = col_start; j < col_end; ++j) {
                            b_data[j] -= lr * gb_data[j];
                        }
                        });
                }
                for (auto& th : threads) th.join();
            }
        }
    };
}


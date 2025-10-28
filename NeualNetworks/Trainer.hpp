#pragma once
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include "Sequential.hpp"
#include "Loss.hpp"
#include "Timer.hpp"
#include "TrainUpdater.hpp" // Updater를 사용하므로 포함

namespace vsnn {
    struct TrainConfig {
        int epochs = 50;
        int batch = 64;
        float lr = 1e-2f;
        int warmup = 1;
        int repeats = 3;
        unsigned seed = 0;
    };

    struct TrainReport {
        double median_ms_per_epoch = 0.0;
        double median_update_ms_per_epoch = 0.0;
        float last_loss = 0.0f;
    };

    class Trainer {
    public:
        template<typename Updater>
        static TrainReport Train(Sequential& model, const Matrix& X, const std::vector<int>& y, const TrainConfig& cfg) {
            SoftmaxCrossEntropy CE;
            Matrix logits, dlogits;
            Timer T, TU;
            std::mt19937 rng(cfg.seed);
            std::vector<double> epoch_ms_list, update_ms_list;
            float last_loss = 0.0f;

            std::vector<int> all_indices(X.rows());
            std::iota(all_indices.begin(), all_indices.end(), 0);

            for (int r = 0; r < cfg.repeats; ++r) {
                std::shuffle(all_indices.begin(), all_indices.end(), rng);

                double sum_epoch_ms = 0.0, sum_up_ms = 0.0;

                for (int e = 0; e < cfg.epochs; ++e) {
                    T.Tic();
                    const auto N = X.rows();
                    for (Eigen::Index beg = 0; beg < N; beg += cfg.batch) {
                        const auto end = std::min(N, beg + cfg.batch);
                        const auto batch_size = end - beg;

                        Matrix Xb(batch_size, X.cols());
                        std::vector<int> yb(batch_size);

                        for (Eigen::Index i = 0; i < batch_size; ++i) {
                            size_t index = static_cast<size_t>(beg) + i;
                            int original_index = all_indices[index];
                            Xb.row(i) = X.row(original_index);
                            yb[i] = y[original_index];
                        }

                        model.Forward(Xb, logits);
                        last_loss = CE.Forward(logits, yb);
                        CE.Backward(yb, dlogits);
                        model.ZeroGrad();
                        model.Backward(dlogits);

                        TU.Tic();
                        Updater::Update(model, cfg.lr);
                        sum_up_ms += TU.TocMs();
                    }
                    double ep_ms = T.TocMs();
                    if (e >= cfg.warmup) sum_epoch_ms += ep_ms;
                }

                int eff_epochs = std::max(0, cfg.epochs - cfg.warmup);
                epoch_ms_list.push_back((eff_epochs > 0) ? (sum_epoch_ms / eff_epochs) : 0.0);
                update_ms_list.push_back((eff_epochs > 0) ? (sum_up_ms / eff_epochs) : 0.0);
            }
            auto median_of = [](std::vector<double>& v) {
                if (v.empty()) return 0.0;
                std::sort(v.begin(), v.end());
                return v[v.size() / 2];
                };
            return { median_of(epoch_ms_list), median_of(update_ms_list), last_loss };
        }
    };
}
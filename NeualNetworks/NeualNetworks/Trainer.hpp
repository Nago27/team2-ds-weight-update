// =============================
// include/vsnn/Trainer.hpp (Vector-based)
// =============================
#pragma once
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <cstring> // memcpy
#include "Sequential.hpp"
#include "Loss.hpp"
#include "Timer.hpp"

using namespace std;

namespace vsnn {
	struct TrainConfig {
		int epochs = 50;
		int batch = 64; // 미니배치
		float lr = 1e-2f;
		int warmup = 1; // 워밍업 epoch (측정 제외)
		int repeats = 3; // 중앙값 산출을 위한 반복 학습 횟수
		unsigned seed = 0;
	};


	struct TrainReport {
		double median_ms_per_epoch = 0.0; // 전체 에폭 평균(워밍업 제외)
		double median_update_ms_per_epoch = 0.0; // 업데이트 구간 합(워밍업 제외)
		float last_loss = 0.0f;
	};

	class Trainer {
	private:
		// 배치 복사 비용 줄이기 (메모리 이동 최적화) <- replace
		static void SliceBatchFast(const Matrix& Xs, const vector<int>& ys,
			int beg, int end, Matrix& Xb, vector<int>& yb)
		{
			const int N = end - beg, D = Xs.Cols();
			if (Xb.Rows() != N || Xb.Cols() != D) Xb.ResetNoInit(N, D);
			yb.resize(N);
			// 행 단위 memcpy (연속 메모리 가정: row-major)
			const size_t row_bytes = static_cast<size_t>(D) * sizeof(float);
			for (int i = 0; i < N; ++i) {
				const float* src = &Xs.Raw()[static_cast<size_t>(beg + i) * D];
				float* dst = &Xb.Raw()[static_cast<size_t>(i) * D];
				memcpy(dst, src, row_bytes);
				yb[i] = ys[beg + i];
			}
		}
	public:
		template<typename Updater = TrainUpdater>
		static TrainReport Train(Sequential& model, const Matrix& X, const vector<int>& y, const TrainConfig& cfg) {
			SoftmaxCrossEntropy CE; Matrix logits, dlogits; Timer T, TU;
			mt19937 rng(cfg.seed);
			vector<double> epoch_ms_list, update_ms_list;
			float last_loss = 0.0f;


			for (int r = 0; r < cfg.repeats; ++r) {
				// 셔플 인덱스
				vector<int> idx(X.Rows());
				iota(idx.begin(), idx.end(), 0);
				shuffle(idx.begin(), idx.end(), rng);


				// 셔플 데이터 복사 (간단 버전)
				Matrix Xs(X.Rows(), X.Cols()); vector<int> ys = y;
				for (int i = 0; i < X.Rows(); ++i) {
					for (int d = 0; d < X.Cols(); ++d) Xs(i, d) = X(idx[i], d);
					ys[i] = y[idx[i]];
				}


				double sum_epoch_ms = 0.0; // 전체 에폭 시간 합
				double sum_up_ms = 0.0; // 업데이트 시간만 합

				Matrix Xb;
				vector<int> yb;

				for (int e = 0; e < cfg.epochs; ++e) {
					T.Tic();
					const int N = Xs.Rows();
					for (int beg = 0; beg < N; beg += cfg.batch) {
						const int end = min(N, beg + cfg.batch);
						// REPLACE: SliceBatch -> SliceBatchFast
						SliceBatchFast(Xs, ys, beg, end, Xb, yb);


						// FWD -> LOSS -> BWD
						model.Forward(Xb, logits);
						last_loss = CE.Forward(logits, yb);
						CE.Backward(yb, dlogits);
						model.ZeroGrad();
						model.Backward(dlogits);

						TU.Tic();
						Updater::Update(model, cfg.lr); // 
						double up_ms = TU.TocMs();
						sum_up_ms += up_ms;
					}
					double ep_ms = T.TocMs();
					if (e >= cfg.warmup) sum_epoch_ms += ep_ms; // 워밍업 제외
				}


				int eff_epochs = max(0, cfg.epochs - cfg.warmup);
				double avg_ep_ms = (eff_epochs > 0) ? (sum_epoch_ms / eff_epochs) : 0.0;
				double avg_up_ms = (eff_epochs > 0) ? (sum_up_ms / eff_epochs) : 0.0;
				epoch_ms_list.push_back(avg_ep_ms);
				update_ms_list.push_back(avg_up_ms);
			}

			auto median_of = [](vector<double>& v) { sort(v.begin(), v.end()); return v[v.size() / 2]; };
			return { median_of(epoch_ms_list), median_of(update_ms_list), last_loss };
		}
	};
}

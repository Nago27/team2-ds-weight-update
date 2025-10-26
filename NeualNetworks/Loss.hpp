// =============================
// include/vsnn/Loss.hpp
// =============================
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include "Matrix.hpp"
#include "Ops.hpp"

using namespace std;

namespace vsnn {
	class SoftmaxCrossEntropy {
	/*
	private:
		Matrix probs_;
	public:
		float Forward(const Matrix& logits, const vector<int>& y) {
			const int N = logits.Rows();
			const int C = logits.Cols();
			if (probs_.Rows() != N || probs_.Cols() != C) probs_.Reset(N, C);
			float loss = 0.0f;
			for (int n = 0; n < N; ++n) {
				Ops::SoftmaxRow(&logits.Data()[static_cast<size_t>(n) * C], &probs_.Data()[static_cast<size_t>(n) * C], C);
				const int t = y[n];
				const float p = max(1e-12f, probs_(n, t));
				loss += -log(p);
			}
			return loss / static_cast<float>(N);
		}
		void Backward(const vector<int>& y, Matrix& dLogits) {
			const int N = probs_.Rows();
			const int C = probs_.Cols();
			if (dLogits.Rows() != N || dLogits.Cols() != C) dLogits.Reset(N, C);
			// dLogits = probs
			for (size_t i = 0; i < probs_.Raw().size(); ++i) dLogits.Raw()[i] = probs_.Raw()[i];
			for (int n = 0; n < N; ++n) dLogits(n, y[n]) -= 1.0f;
			const float invN = 1.0f / static_cast<float>(N);
			for (auto& v : dLogits.Raw()) v *= invN;
		}
		*/
	public:
		// Fused: computes loss and writes dLogits = (softmax - onehot)/N
		float ForwardBackward(const Matrix& logits, const std::vector<int>& y, Matrix& dLogits) {
			const int N = logits.Rows();
			const int C = logits.Cols();
			if (dLogits.Rows() != N || dLogits.Cols() != C) dLogits.Reset(N, C);
			float loss = 0.0f;
			for (int n = 0; n < N; ++n) {
				const float* lrow = &logits.Raw()[(size_t)n * C];
				float* drow = &dLogits.Raw()[(size_t)n * C];
				Ops::SoftmaxRow(lrow, drow, C); // write probs into drow
				const int t = y[n];
				const float p = std::max(1e-12f, drow[t]);
				loss += -std::log(p);
				drow[t] -= 1.0f; // drow = probs - onehot
			}
			const float invN = 1.0f / std::max(1, N);
			for (auto& v : dLogits.Raw()) v *= invN;
			return loss * invN;
		}

		// Back-compat (not used in patched Trainer)
		float Forward(const Matrix& logits, const std::vector<int>& y) {
			Matrix tmp; return ForwardBackward(logits, y, tmp);
		}
		void Backward(const std::vector<int>& /*y*/, Matrix& /*dLogits*/) {}
	};
}

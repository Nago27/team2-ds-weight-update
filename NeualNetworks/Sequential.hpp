// =============================
// include/vsnn/Sequential.hpp
// =============================
#pragma once
#include <memory>
#include <vector>
#include "Layer.hpp"

using namespace std;

namespace vsnn {
	class Sequential {
	private:
		vector<unique_ptr<Layer>> layers_;
		vector<Matrix> acts_; // X0..X_L
		vector<Matrix> d_acts_; //dX_L..dX0
	public:
		template<typename T, typename... Args>
		T* Add(Args&&... args) {
			layers_.push_back(make_unique<T>(forward<Args>(args)...));
			return static_cast<T*>(layers_.back().get());
		}
		void Forward(const Matrix& X, Matrix& out) {
			acts_.resize(layers_.size() + 1);
			acts_[0] = X;
			
			for (size_t i = 0; i < layers_.size(); ++i) {
				layers_[i]->Forward(acts_[i], acts_[i+1]);
			}
			out = acts_.back();
		}
		void Backward(const Matrix& dOut) {
			d_acts_.resize(acts_.size());
			d_acts_.back() = dOut;
			for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
				layers_[i]->Backward(acts_[i], d_acts_[i+1], d_acts_[i], i);
			}
		}
		void ZeroGrad() { for (auto& L : layers_) L->ZeroGrad(); }
		void Step(float lr) { for (auto& L : layers_) L->Step(lr); }
		// ---- Introspection for Updater ----
		size_t NumLayers() const { return layers_.size(); }
		Layer* LayerAt(size_t i) { return layers_[i].get(); }
		const Layer* LayerAt(size_t i) const { return layers_[i].get(); }
	};
}
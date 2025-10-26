// =============================
// include/vsnn/Activations.hpp
// =============================
#pragma once
#include "Layer.hpp"
#include "Ops.hpp"


namespace vsnn {
	class ReLU : public Layer {
		std::vector<int> mask_index_flat_; // stores r*C + c of positive activations
	public:
		void Forward(const Matrix& X, Matrix& Y) override {
			Ops::ReLUForwardWithMask(X, Y, mask_index_flat_);
		}
		void Backward(const Matrix& /*X*/, const Matrix& dY, Matrix& dX) override {
			Ops::ReLUBackwardWithMask(mask_index_flat_, dY, dX);
		}
		/*
		void Forward(const Matrix& X, Matrix& Y) override { Ops::ReLUForward(X, Y); }
		void Backward(const Matrix& X, const Matrix& dY, Matrix& dX) override { Ops::ReLUBackward(X, dY, dX); }
		*/
	};
}
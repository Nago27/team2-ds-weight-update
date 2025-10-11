// =============================
// include/vsnn/Matrix.hpp
// =============================
#pragma once
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
using namespace std;

namespace vsnn {
	using f32 = float;
	using i32 = int32_t;


	class Matrix {
	private:
		i32 rows_ = 0, cols_ = 0;
		vector<f32> data_;
		mutable std::unique_ptr<Matrix> transpose_cache_;
		mutable bool transpose_dirty_ = true;
	public:
		Matrix() = default;
		Matrix(i32 r, i32 c) { Reset(r, c); }
		void Reset(i32 r, i32 c) {
			rows_ = r; cols_ = c;
			data_.assign((size_t)r * c, 0.0f);
			transpose_cache_.reset();
			transpose_dirty_ = true;
		}
		inline i32 Rows() const { return rows_; }
		inline i32 Cols() const { return cols_; }
		inline f32* Data() { return data_.data(); }
		inline const f32* Data() const { return data_.data(); }
		inline f32& operator()(i32 r, i32 c) { return data_[static_cast<size_t>(r) * cols_ + c]; }
		inline f32 operator()(i32 r, i32 c) const { return data_[static_cast<size_t>(r) * cols_ + c]; }
		inline void Fill(f32 v) { fill(data_.begin(), data_.end(), v); }
		inline const vector<f32>& Raw() const { return data_; }
		inline vector<f32>& Raw() { return data_; }

		//전치행렬
		const Matrix& Transposed() const {
			if (!transpose_cache_ || transpose_dirty_) {
				transpose_cache_ = std::make_unique<Matrix>(cols_, rows_);
				Matrix& T = *transpose_cache_;
				for (i32 i = 0; i < rows_; ++i) {
					const f32* src = &data_[(size_t)i * cols_];
					for (i32 j = 0; j < cols_; ++j)
						T(j, i) = src[j];
				}
				transpose_dirty_ = false;
			}
			return *transpose_cache_;
		}

		Matrix(const Matrix& other) {
			rows_ = other.rows_;
			cols_ = other.cols_;
			data_ = other.data_;
			transpose_cache_.reset();
			transpose_dirty_ = true;
		}
		Matrix& operator=(const Matrix& other) {
			if (this == &other) return *this;
			rows_ = other.rows_;
			cols_ = other.cols_;
			data_ = other.data_;
			transpose_cache_.reset();
			transpose_dirty_ = true;
			return *this;
		}
		void MarkDirty() { transpose_dirty_ = true; }
	};
}

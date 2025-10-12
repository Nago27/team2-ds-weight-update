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
	
	//블록 사이즈 설정 (CPU마다 최적값이 다르므로 32, 64, 128 등 2^n으로 변경해가며 실행해봐야 함)
	const int BLOCK = 128;

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

		//전치행렬(처음 생성될 때나 업데이트 시에만 기존 행렬에서 전치 행렬로 복사, 그 외에는 기존 전치행렬 반환만)
		const Matrix& Transposed() const {
			if (!transpose_cache_ || transpose_dirty_) {
				transpose_cache_ = std::make_unique<Matrix>(cols_, rows_);
				Matrix& T = *transpose_cache_;
				int M = rows_, K = cols_;

			#ifdef _OPENMP
				#pragma omp parallel for
			#endif 
				for (i32 i0 = 0; i0 < rows_; i0 += BLOCK) {
					int iMax = std::min(i0 + BLOCK, M);
					for (i32 j0 = 0; j0 < cols_; j0 += BLOCK) {
						int jMax = std::min(j0 + BLOCK, K);
						for (i32 i = i0; i < iMax; ++i) {
							const f32* src = &data_[(size_t)i * cols_];
							for (i32 j = j0; j < jMax; ++j) T(j, i) = src[j];
						}
					}
				}

				transpose_dirty_ = false;
			}
			return *transpose_cache_;
		}
		void MarkDirty() { transpose_dirty_ = true; }
		
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
		
	};
}

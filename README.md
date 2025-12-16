# 자료구조실습 2조 
<div align="left">2022203031 강은우</div>
<div align="left">2022203050 김건우</div>
<div align="left">2022203092 이동현</div>
<div align="left">2022203036 임동건</div>

## 1. 개발 환경 설정
### 1) OpenMP & AVX2 활성화
- C/C++ > 언어 > OpenMP 지원 > 예(/openmp)
- C/C++ > 코드 생성 > 고급 명령 집합 사용 > 고급 벡터 확장 2(X86/X64)(/arch:AVX2)

### 2) 프로젝트 속성 변경 절차
- C/C++ > 최적화 > 최대 최적화(속도 우선)(/O2)
- C/C++ > 최적화 > 전체 프로그램 최적화 > 예(/GL)
- C/C++ > 코드 생성 > 기본 런타임 검사 > 기본값
- C/C++ > 일반 > 디버그 정보 형식 > 프로그램 데이터베이스(/Zi)

## 2. 주요 구현 소스코드
- 행 연산 최적화
```cpp
// [Ops.hpp] MatMul1: Y = X * W 최적화 구현
static void MatMul1(const Matrix& A, const Matrix& B, Matrix& C) {
    int M = A.Rows(), K = A.Cols(), N = B.Cols();
    C.Reset(M, N);
    
    // 행렬 크기(K, N)에 따른 루프 구조 분기
    if (K < N) {
#pragma omp parallel for // OpenMP 병렬화
        for (int i = 0; i < M; ++i) {
            const float* a = &A.Raw()[(size_t)i * K];
            float* c = &C.Raw()[(size_t)i * N];
            for (int j = 0; j < K; ++j) {
                // Sparsity 활용 (0인 값 연산 생략)
                if (a[j] == 0.0f) continue;

                const __m256 a_vec = _mm256_set1_ps(a[j]);
                const float* b = &B.Raw()[(size_t)j * N];
                int k = 0;
                // AVX2 SIMD 병렬 연산 (8 float 처리)
                for (; k + 8 <= N; k += 8) {
                    __m256 b_vec = _mm256_loadu_ps(b + k);
                    __m256 c_vec = _mm256_loadu_ps(c + k);
                    c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                    _mm256_storeu_ps(c + k, c_vec);
                }
                for (; k < N; k++) c[k] += a[j] * b[k]; // 나머지 처리
            }
        }
    } 
    else {
        // Row-major 연속 접근 유지를 위한 전치 행렬(Transpose) 생성
        // B 행렬의 열 접근(Cache Miss 유발)을 행 접근으로 변환
        Matrix BT(N, K);
        #pragma omp parallel for
        for (i32 i = 0; i < K; ++i) {
            const f32* src = &B.Raw()[(size_t)i * N];
            for (i32 j = 0; j < N; ++j) BT(j, i) = src[j];
        }

        #pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            const float* a = &A.Raw()[(size_t)i * K];
            float* c = &C.Raw()[(size_t)i * N];
            for (int j = 0; j < N; ++j) {
                __m256 sum_vec = _mm256_setzero_ps();
                const float* b = &BT.Raw()[(size_t)j * K]; // 연속 메모리 접근(Cache Hit)
                int k = 0;
                for (; k + 8 <= K; k += 8) {
                    __m256 a_vec = _mm256_loadu_ps(a + k);
                    __m256 b_vec = _mm256_loadu_ps(b + k);
                    sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
                }
                // ... (Reduction 및 나머지 처리 생략)
            }
        }
    }
}
```
- 메모리 복사 최적화 (Sequential & SliceBatch)
```cpp
// [Sequential.hpp] 
// 불필요한 메모리 복사 제거
void Forward(const Matrix& X, Matrix& out) {
  acts_.resize(layers_.size() + 1);
  acts_[0] = X;

  for (size_t i = 0; i < layers_.size(); ++i) {
    layers_[i]->Forward(acts_[i], acts_[i + 1]);
  }
  out = acts_.back();
}
void Backward(const Matrix& dOut) {
  d_acts_.resize(acts_.size());
  d_acts_.back() = dOut;
  for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
    layers_[i]->Backward(acts_[i], d_acts_[i + 1], d_acts_[i], i);
  }
}

// [Trainer.hpp] SliceBatch
static void SliceBatch(const Matrix& X, const vector<int>& y, const vector<int>& idx, int beg, int end, Matrix& Xb, vector<int>& yb) {
    const int N = end - beg; const int D = X.Cols();
    if (Xb.Rows() != N || Xb.Cols() != D) Xb.Reset(N, D);
    yb.resize(N);
    float* xb = &Xb.Raw()[0];
    const float* x = &X.Raw()[0];

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        int actual_row_idx = idx[beg + i];
        const float* src_row = x + (size_t)actual_row_idx * D;
        float* dst_row = xb + (size_t)i * D;
        // [Optimization] memcpy로 행 단위 고속 복사
        memcpy(dst_row, src_row, D * sizeof(float));
        yb[i] = y[actual_row_idx];
    }
}
```
- 불필요한 dX 연산 삭제
```cpp
// [Dense.hpp] Backward
void Backward(const Matrix& X, const Matrix& dY, Matrix& dX, int i) override {
    // 1. gW, gb 계산 (생략)
    Ops::MatMul2(X, dY, gW_);
    
    // ... (gb 계산 및 병렬화 로직)

    // 2. dX (입력 기울기) 계산 최적화
    // 입력층(i=0)일 경우 이전 층으로 전파할 필요가 없으므로 연산 생략
    if (i != 0) {
        Ops::MatMul3(dY, W_, dX);
    }
}
```

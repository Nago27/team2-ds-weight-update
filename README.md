# 자료구조실습 2조 
<div align="left">2022203031 강은우</div>
<div align="left">2022203050 김건우</div>
<div align="left">2022203092 이동현</div>
<div align="left">2022203036 임동건</div>

## 문제점 파악
### 행 우선 접근 방식
- Matrix.hpp에서는 행 우선(row_major) 저장 방식이지만, Ops.hpp, Dense.hpp 에서는 **열 단위**로 접근하여 캐시 미스가 발생합니다.
```cpp
// vector와 operator를 이용해 행 우선 연속 저장을 하는 Matrix
class Matrix {
	private:
		i32 rows_ = 0, cols_ = 0;
		vector<f32> data_;
	public:
		inline f32 operator()(i32 r, i32 c) const { return data_[static_cast<size_t>(r) * cols_ + c]; }
	};
```
```cpp
// W에 열 단위로 접근하는 연산
static void MatMul(const Matrix& X, const Matrix& W, Matrix& Y) {
			assert(X.Cols() == W.Rows());
			if (Y.Rows() != X.Rows() || Y.Cols() != W.Cols()) Y.Reset(X.Rows(), W.Cols());
			for (i32 n = 0; n < X.Rows(); ++n) {
				for (i32 j = 0; j < W.Cols(); ++j) {
					float acc = 0.0f;
					for (i32 k = 0; k < X.Cols(); ++k) acc += X(n, k) * W(k, j);
					Y(n, j) = acc;
				}
			}
		}
```
### 불필요한 연산/복사 비용
- Trainer.hpp의 Train과 SliceBatch에서 배치 구성시에 깊은 복사가 매 스텝 마다 발생되므로 복사량이 많이 누적되어 시간/메모리 대역폭을 낭비하게 됩니다.
```cpp
// Train에서 셔플된 인덱스로 전체 데이터를 복사 하고 SliceBatch로 넘겨 함수 내부에서 한번 더 복사 수행
Matrix Xs(X.Rows(), X.Cols()); vector<int> ys = y;
for (int i = 0; i < X.Rows(); ++i) {
	for (int d = 0; d < X.Cols(); ++d) Xs(i, d) = X(idx[i], d);
	ys[i] = y[idx[i]];
}

SliceBatch(Xs, ys, beg, end, Xb, yb);
```
```cpp
static void SliceBatch(const Matrix& X, const vector<int>& y, int beg, int end, Matrix& Xb, vector<int>& yb) {
    const int N = end - beg; const int D = X.Cols();
    // 매번 메모리를 재할당하거나 체크함
    if (Xb.Rows() != N || Xb.Cols() != D) Xb.Reset(N, D);
    yb.resize(N);
    
    // [문제점] 이중 루프를 돌며 원본 데이터를 또 다른 메모리 공간(Xb)으로 '깊은 복사' 수행
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d) Xb(i, d) = X(beg + i, d); // 값 복사 발생
        yb[i] = y[beg + i];
    }
}
```
- Sequential.hpp의 forward(), backward()에서 레이어 간 데이터 전달시 전체 데이터의 깊은 복사를 반복적으로 수행하고 있어서 실제 연산보다 메모리 상에서 이동시키는 데 과도한 시간이 소요됩니다.
```cpp
void Forward(const Matrix& X, Matrix& out) {
   acts_.resize(layers_.size() + 1);
   acts_[0] = X;
   Matrix cur = X, nxt;
   for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->Forward(cur, nxt);
      acts_[i + 1] = nxt;
      cur = acts_[i + 1];
   }
   out = acts_.back();
}
void Backward(const Matrix& dOut) {
   Matrix cur_d = dOut, prev_d;
   for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
      layers_[i]->Backward(acts_[i], cur_d, prev_d);
      cur_d = prev_d;
   }
}
```
- 불필요한 연산 dX
  현재 역전파 과정에서 Dense 레이어는 입력 데이터에 대한 기울기 dX를 계산하는데, 맨 처음 입력층에서 계산된 dX는 이전 단계가 없으므로 사용되지 않고 버려집니다.
```cpp
// [Dense.hpp] void Backward(...) 내부
if (dX.Rows() != X.Rows() || dX.Cols() != W_.Rows()) dX.Reset(X.Rows(), W_.Rows());
for (i32 i = 0; i < X.Rows(); ++i) {
    for (i32 k = 0; k < W_.Rows(); ++k) {
        float acc = 0.0f; 
        for (i32 j = 0; j < W_.Cols(); ++j) acc += dY(i, j) * W_(k, j); // 불필요한 연산
        dX(i, k) = acc;
    }
}

// [Sequential.hpp]
void Backward(const Matrix& dOut) {
    Matrix cur_d = dOut, prev_d;
    // 레이어를 거꾸로 타고 올라감
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        // i=0 (입력층)일 때도 prev_d(dX)를 계산함.
        layers_[i]->Backward(acts_[i], cur_d, prev_d);
        cur_d = prev_d; // i=0일 때 계산된 cur_d는 루프 종료 후 버려짐 (낭비)
    }
}
```

## 문제를 해결하기 위한 자료구조    
- 기존 Matrix(row_major)의 사용 방식을 행 단위 연산으로 변경 (자료구조 활용 변경)

## 주요 구현 내용
#### 행 연산 변경 및 OpenMP & AVX2 적용 (작성중)
1. MatMul 함수 분리:
   <br>행렬곱 연산마다 행렬의 전치 형태가 다르다는 점을 고려하여 MatMul 함수를 3가지로 분리하였습니다.
   - ```Ops::MatMul1```: $Y = X \times W$ (행 누적 + 전치 행렬 + 희소성 데이터 스킵)
   - ```Ops::MatMul2```: $gW = X^T \times dY$ (행 누적 + 전치 행렬 + 희소성 데이터 스킵)
   - ```Ops::MatMul3```: $dX = dY \times W^T$ (행 누적 + 전치 행렬)
3. 루프 구조 최적화 (병렬처리): 
   <br>레이어마다 행렬의 크기가 다르다는 것을 고려하여 각 함수의 내부에서도 if문으로 분기를 만들어 총 5가지의 루프를 구현하였습니다.<br>
   각 루프의 순서는 OpenMP(스레드 병렬)/AVX2(SIMD 병렬)를 활용한 병렬화 효율과 스레드 간의 Race Condition을 고려하여 결정하였습니다.
4. Row-major 연속 접근 유지로 캐시 효율 극대화
   <br>: 루프 순서를 바꾸는 과정에서 열 단위 접근이 발생하는 행렬은 함수 내부에서 전치 행렬을 생성하여 연산에 활용함으로써, 비효율적인 열 단위 접근을 행 단위 접근으로 변환하였습니다.
   결과적으로 루프의 순서가 바뀌더라도 Row_major로 저장된 연속적인 자료구조를 순차적으로 읽을 수 있게 되어 메모리의 연속성을 극대화하고 캐시 미스를 최소화하였습니다.
5. 희소성 활용
   <br>: 연산 과정에서 데이터가 0인 경우 연산을 생략(```continue```)하는 방식으로 데이터의 희소성을 활용하였습니다.
6. 기타 연산 최적화
   <br>: ```AddRowBias```나 ```ReLUForward``` 등 다른 모든 행렬 연산에서도 행 단위 접근을 극대화하고 OpenMP, AVX2를 적절히 사용하였습니다.
7. OpenMP & AVX2 적용 방법
- C/C++ > 언어 > OpenMP 지원 > 예(/openmp)
- C/C++ > 코드 생성 > 고급 명령 집합 사용 > 고급 벡터 확장 2(X86/X64)(/arxh:AVX2)
- #include <immintrin.h>

#### 불필요한 메모리 복사 최적화
1. Sequential 클래스의 데이터 전달 구조 개선
  - Forward  최적화: 벡터의 요소를 직접 참조하여 다음 레이어의 입력과 출력으로 사용하도록 변경함으로써, 불필요한 행렬 복사를 방지하였습니다.
  - Backward 최적화: 역전파 시에도 벡터를 도입하여 미분값 행렬을 별도의 복사 없이 해당 메모리 주소에 직접 기록하도록 개선하였습니다.
2. Trian과 SliceBatch 메모리 처리 효율화 (memcpy 및 병렬화 적용)
  - Train에서 셔플된 인덱스로 전체 데이터를 복사해서 넘기는 것이 아닌 셔플된 인덱스를 SliceBatch로 넘겨서 함수 내부에서 한번만 복사하는 방식으로 수정하였습니다.
  - 기존의 이중 루프를 통한 대입 방식을 memcpy를 활용한 행 단위 블록 메모리 복사로 변경하여 대입 연산 속도를 최적화하였습니다.
  - 배치 생성 과정을 멀티스레드로 병렬화함으로써, 대용량 데이터 복사 시의 처리량을 증대시켰습니다.

#### dX 연산 삭제 (Dense.hpp)
Dense 레이어의 Backward 함수가 현재 자신이 몇 번째 레이어인지 알 수 있도록 인덱스($i$)를 인자로 받게 수정하였습니다. <br>
이를 통해 현재 레이어가 입력층($i=0$)인 경우, 무거운 행렬 곱셈 연산인 $dX$ 계산 과정을 아예 **생략**하도록 조건문을 추가하였습니다.
```cpp
// 수정된 Backward 함수: 레이어 인덱스 'i'를 매개변수로 추가
void Backward(const Matrix& X, const Matrix& dY, Matrix& dX, int i) override {
    // 가중치(gW) 및 편향(gb) 기울기는 정상적으로 업데이트
    Ops::MatMul2(X, dY, gW_);

    // dX (입력에 대한 기울기) 계산 최적화
    // 입력층(i=0)일 경우, 이전 층으로 전파할 오차가 없으므로 계산을 수행하지 않음.
    if (i != 0) {
        Ops::MatMul3(dY, W_, dX);
    }
}
```

#### 프로젝트 속성 변경 (작성중)
- C/C++ > 최적화 > 최대 최적화(속도 우선)(/O2)
- C/C++ > 최적화 > 전체 프로그램 최적화 > 예(/GL)
- C/C++ > 코드 생성 > 기본 런타임 검사 > 기본값
- C/C++ > 일반 > 디버그 정보 형식 > 프로그램 데이터베이스(/Zi)

/O2와 /GL 옵션을 적용하여 컴파일 및 링크 단계 최적화를 적용하였고 기본 런타임 검사를 기본값으로 바꿔주어 실행 중 오버헤드를 제거하였습니다. /Zi는 실행 속도에는 영향을 주지 않으나 /GL 사용에 따른 옵션 의존성으로 인해 적용하였습니다.

## 실행 결과 (전/후 훈련시간 비교)
### 행 단위 연산 변경
- Before
<img width="631" height="550" alt="Image" src="https://github.com/user-attachments/assets/5629467b-e76c-4bbe-b975-99c4bf3c70e7" />

- After
<img width="620" height="539" alt="Image" src="https://github.com/user-attachments/assets/7e366b91-7e33-4f6a-9e67-fefcac4804e0" />

### 불필요한 dX 연산 삭제/메모리 복사 최적화
- Before
<img width="620" height="539" alt="Image" src="https://github.com/user-attachments/assets/7e366b91-7e33-4f6a-9e67-fefcac4804e0" />

- After
<img width="632" height="542" alt="Image" src="https://github.com/user-attachments/assets/742b4b2f-e863-4ed1-bfcf-ea0187835bf9" />

### OpenMP, AVX2
- Before
<img width="632" height="542" alt="Image" src="https://github.com/user-attachments/assets/742b4b2f-e863-4ed1-bfcf-ea0187835bf9" />

- After
<img width="621" height="554" alt="Image" src="https://github.com/user-attachments/assets/11c9f10d-f802-4a62-90f5-fb6888d1e247" />

### 프로젝트 속성 변경
- Before
<img width="621" height="554" alt="Image" src="https://github.com/user-attachments/assets/11c9f10d-f802-4a62-90f5-fb6888d1e247" />

- After
<img width="618" height="545" alt="Image" src="https://github.com/user-attachments/assets/b841c0bb-4096-4f22-abc0-28a0d7716b22" />

## 팀원들의 역할
- 강은우(조장): 자료조사, 행 단위 연산 변경 구현
- 김건우: 자료조사(Eigen 외부 라이브러리 분석), 메모리 복사 최적화
- 이동현: 자료조사(Eigen 외부 라이브러리 분석), OpenMP 및 AVX2 적용 및 구현
- 임동건: 자료조사, GitHub 협업 개발 환경 구축, 중간발표 PPT 및 최종보고서 작성
  
## 진행 과정 및 일정
- 3~5주차: 자료조사
- 6~11주차: 행 연산 변경, 데이터 복사 최적화
- 12~14주차: OpenMP와 AVX2 적용, 프로젝트 속성 변경
- 15주차: 최종 보고서 작성

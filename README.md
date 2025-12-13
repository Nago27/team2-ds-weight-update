# 자료구조실습 2조 
<div align="left">2022203031 강은우</div>
<div align="left">2022203050 김건우</div>
<div align="left">2022203092 이동현</div>
<div align="left">2022203036 임동건</div>

## 문제점 파악
### 행 우선 접근 방식
- Matrix.hpp에서는 행 우선(row_major) 접근 방식이지만, Ops.hpp, Dense.hpp 에서는 ‘열‘ 방향으로 접근하여 캐시 미스가 발생
- $gW = X^T \times dY$, $gb = ∑rows(dy)$ 계산시 **열 방향**으로 누적되어 비연속 접근이 반복되고 있음

### 불필요한 연산/복사 비용
- SliceBatch에서 현재 배치 구성시에 깊은 복사가 매 스텝 마다 발생되므로 복사량이 많이 누적되어 시간/메모리 대역폭을 잡아먹음
- Sequential.hpp의 forward(), backward()에서 불필요한 복사 과정이 있다.
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
```cpp
```

## 문제를 해결하기 위한 자료구조    
- 기존 Matrix(row_major)의 사용 방식을 행 단위 연산으로 변경 (자료구조 활용 변경)

## 주요 구현 내용
#### 행 연산 변경 및 OpenMP & AVX 적용
- Ops::MatMul1: $Y = X \times W$ (행 누적 + 희소성 데이터 스킵)
- Ops::MatMul2: $gW = X^T \times dY$ (행 누적 + 희소성 데이터 스킵)
- Ops::MatMul3: $dX = dY \times W^T$ (전치 행렬)
- AddRowBias 수정
- ReLU Forward/Backward 연산

시간을 가장 많이 차지하는 행렬곱 연산을 Row_major 방식으로 저장된 자료구조의 특성을 가장 잘 활용할 수 있는 구조로 바꾸었다. 먼저 행렬곱 연산마다 행렬의 전치 형태($A \times B$, $A^T \times B$ 등)가 다르다는 점을 고려하여 MatMul 함수를 3가지로 분리하였다. 또한 레이어마다 행렬의 크기가 다르다는 것을 고려하여 각 함수의 내부에서도 if문으로 분기를 만들어 총 5가지의 루프를 구현하였다. 각 루프의 순서는 OpenMP(스레드 병렬)/AVX2(SIMD 병렬)를 활용한 병렬화 효율과 스레드간의 race condition을 고려하여 결정하였다. 루프 순서를 바꾸는 과정에서 열 단위 접근이 발생하는 행렬은 함수 내부에서 전치 행렬을 생성하여 연산에 활용함으로써 비효율적인 열 단위 접근을 행 단위 접근으로 변환하였다. 결과적으로 루프의 순서가 바뀌더라도 Row_major로 저장된 연속적인 자료구조를 순차적으로 읽을 수 있게 되어 메모리의 연속성을 극대화하고 캐시 미스를 최소화하였다.
추가적으로 연산 과정에서 데이터가 0인 경우 연산을 생략하는 방식으로 데이터의 희소성도 활용하였다.
AddRowBias나 ReLUForward등 다른 모든 행렬 연산에서도 행 단위 접근을 극대화하고 OpenMP, AVX2를 적절히 사용하였다.

#### 불필요한 메모리 복사 최적화

#### dX 연산 삭제 (Dense.hpp)

#### 프로젝트 속성 변경


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

### OpenMP, AVX
- Before
<img width="632" height="542" alt="Image" src="https://github.com/user-attachments/assets/742b4b2f-e863-4ed1-bfcf-ea0187835bf9" />

- After
<img width="621" height="554" alt="Image" src="https://github.com/user-attachments/assets/11c9f10d-f802-4a62-90f5-fb6888d1e247" />

### 프로젝트 속성 변경
- Before
<img width="621" height="554" alt="Image" src="https://github.com/user-attachments/assets/11c9f10d-f802-4a62-90f5-fb6888d1e247" />

- After
<img width="625" height="546" alt="Image" src="https://github.com/user-attachments/assets/21b58bfd-5405-4f2c-b5a5-b6cae12e8c79" />

## 팀원들의 역할
- 강은우(조장): 자료조사, 행 단위 연산 변경 구현
- 김건우: 자료조사(Eigen 외부 라이브러리 분석), 메모리 복사 최적화
- 이동현: 자료조사(Eigen 외부 라이브러리 분석), OpenMP 및 AVX 적용 및 구현
- 임동건: 자료조사, GitHub 협업 개발 환경 구축, 중간발표 PPT 및 최종보고서 작성
  
## 진행 과정 및 일정
- 3~5주차: 자료조사
- 6~11주차: 행 연산 변경, 데이터 복사 최적화
- 12~14주차: OpenMP와 AVX 적용, 프로젝트 속성 변경
- 15주차: 최종 보고서 작성

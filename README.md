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
- Ops::MatMul2: $gW = X^T * dY$ (행 누적 + 희소성 데이터 스킵)
- Ops::MatMul3: $dX = dY * W^T$ (전치 행렬)
- AddRowBias 수정
- LeLU Forward/Backward 연산

​행렬 곱 연선 구현은 OpenMP(스레드 병렬)와 AVX2(SIMD 병렬)를 통한 하드웨어 가속, 그리고 연속적인 자료구조의 활용이라는 두 가지 핵심 목표를 동시에 달성하는 데 집중하였다. 먼저, 각 행렬 곱 연산마다 전치 형태($A \times B$ vs $A^T \times B$)가 다르다는 점을 고려하여 MatMul 함수를 3가지로 분리하였다. 이후 하드웨어 효율을 극대화하기 위해 루프 순서를 재설계하였다. 구체적으로는 한 번에 데이터를 묶어 처리하는 AVX2의 성능을 위해 가장 긴 축을 안쪽 루프에 배치하고, OpenMP가 적용되는 바깥쪽 루프에는 충분히 큰 축을 배치하여 유휴 스레드 없이 부하를 균등하게 분산시켰다. 또한 레이어마다 행렬의 크기가 다르다는 점을 반영하여 3개의 MatMul 함수 내부에서도 행렬의 크기 조건에 따라 분기되도록 설계해 총 5가지의 최적화된 루프를 구현하였다. 루프 순서 변경 과정에서 발생하는 열 단위 접근과 멀티스레드 경쟁 상태문제는 함수 내부에서 연산 전에 대상 행렬의 전치행렬을 생성하여 해결하였다. 전치행렬을 이용함으로써 행 단위 접근으로 변환해 자료구조의 물리적 연속성을 극대화하였으며 동시에 각 스레드가 서로 다른 행에만 쓰기 작업을 수행하도록 유도하여 별도의 동기화 비용 없이도 스레드 안전성을 확보하였다. 추가적으로 입력 데이터가 0인 경우 연산을 생략하여 데이터의 희소성을 활용하였으며, 이러한 행 단위 연속 접근 및 AVX2 활용은 AddRowBias와 ReLU Forward/Backward 등 프로젝트의 모든 연산에 적용하였다
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
- After

### OpenMP, AVX
- Before
- After

### 프로젝트 속성 변경
- Before
- After

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

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

#### 불필요한 메모리 복사 최적화

#### dX 연산 삭제 (Dense.hpp)

#### 프로젝트 속성 변경


## 실행 결과 (전/후 훈련시간 비교)
### 행 단위 연산 변경
- Before
- After

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

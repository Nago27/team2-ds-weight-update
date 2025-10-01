# team2-ds-weight-update
(자료구조실습 2조) 신경망 가중치 업데이트 속도 증가

브랜치 전략: **`develop`로 PR**, `main`은 릴리스만.

## 초기세팅
```bash
git config --global user.name "계정이름"
git config --global user.email "email"
```

## 저장소 클론
```bash
git clone https://github.com/Nago27/team2-ds-weight-update.git
cd team2-ds-weight-update.git
git fetch origin
```

## 작업 브랜치 (develop에서 분기)
```bash
git checkout -b 브랜치명 origin/develop
```

## 커밋 & 푸시
```bash
git add NeuralNetworks
git commit -m "(무엇을 변경했는지 한 줄)"
git push -u origin HEAD
```
### 11월 19일 중간 발표 예정

---
## 10.1일 회의 내용
### ➡️ 행렬 연산 속도 최적화로
1. Eigen 라이브러리 사용 -> Matrix.hpp 수정 (동현, 건우)
2. 새로운 자료구조 구현 (행렬 연산 최적화)

---
### 행렬 곱셈 비효율성
불필요한 데이터 복사
```cpp
Matrix Xs(X.Rows(), X.Cols()); vector<int> ys = y;
for (int i = 0; i < X.Rows(); ++i) {
    for (int d = 0; d < X.Cols(); ++d) Xs(i, d) = X(idx[i], d);
    ys[i] = y[idx[i]];
}
```

[3중 반복문]
-> 순수 최적화만 했을 때 약 280초  
Ops.hpp의 MatMul()
```cpp
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

Dense.hpp의 BackWard()
```cpp
void Backward(const Matrix& X, const Matrix& dY, Matrix& dX) override {
    if (gW_.Rows() != W_.Rows() || gW_.Cols() != W_.Cols()) gW_.Reset(W_.Rows(), W_.Cols());
        for (i32 k = 0; k < W_.Rows(); ++k) {
            for (i32 j = 0; j < W_.Cols(); ++j) {
                float acc = 0.0f;
                for (i32 i = 0; i < X.Rows(); ++i) acc += X(i, k) * dY(i, j);
                gW_(k, j) = acc;
            }
        }
    }
}
```

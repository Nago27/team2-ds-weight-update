#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include <array>
#include <numeric>
#include <memory>
#include <chrono> // Timer 클래스를 위해 추가

// Eigen 라이브러리 헤더
#include <Eigen/Dense>

namespace vsnn {
    // 1. Matrix 타입 정의
    using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    //Matrix가 제공하는 행렬 클래스 탬플릿
    //Matrix<행렬의 원소, 행의 개수, 열의 개수, 메모리 저장 순서를 행 우선>
    using f32 = float;
    using i32 = int32_t;

    // 2. Layer 베이스 클래스
    class Layer {
        //Dense, RELU 같은 신경망의 설계도
    public:
        virtual ~Layer() = default;
        virtual void Forward(const Matrix& X, Matrix& Y) = 0; //순전파
        virtual void Backward(const Matrix& X, const Matrix& dY, Matrix& dX) = 0; //역전파
        virtual void ZeroGrad() {} //그래디언트 초기화
        virtual void Step(float lr) {} //가중치 업데이트
    };

    // 3. Sequential 모델 클래스 (완성본)
    class Sequential : public Layer {
    private:
        std::vector<std::shared_ptr<Layer>> layers_;
        std::vector<Matrix> inputs_for_backward_;
    public:
        template<typename T, typename... Args>
        void Add(Args&&... args) {
            layers_.push_back(std::make_shared<T>(std::forward<Args>(args)...));
        }
        void Forward(const Matrix& X, Matrix& Y) override {
            if (layers_.empty()) { Y = X; return; }
            inputs_for_backward_.resize(layers_.size());
            Matrix current_input = X;
            for (size_t i = 0; i < layers_.size(); ++i) {
                inputs_for_backward_[i] = current_input;
                Matrix current_output;
                layers_[i]->Forward(current_input, current_output);
                current_input = current_output;
            }
            Y = current_input;
        }
        void Backward(const Matrix&, const Matrix& dY, Matrix& dX) override {
            if (layers_.empty()) { dX = dY; return; }
            Matrix current_grad = dY;
            Matrix prev_grad;
            for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
                layers_[i]->Backward(inputs_for_backward_[i], current_grad, prev_grad);
                current_grad = prev_grad;
            }
            dX = current_grad;
        }
        void ZeroGrad() override { for (auto& layer : layers_) layer->ZeroGrad(); }
        void Step(float lr) override { for (auto& layer : layers_) layer->Step(lr); }
    };

    // 10. Initializater
    class Initializer {
    public:
        //행렬 W를 균등 분포의 난수로 초기화
        static void Uniform(Matrix& W, float scale = 0.01f, uint64_t seed = 123) {
            // 고정 시드 사용(123)하여 난수 생성기 만듬
            std::mt19937_64 gen(seed);
            //균등 분포의 번위 정의
            std::uniform_real_distribution<float> dist(-scale, scale);

            //행렬 데이터의 시작 주소 포인터
            float* data_ptr = W.data();
            //
            for (int i = 0; i < W.size(); ++i) {
                data_ptr[i] = dist(gen);
            }
        }
    };

    // 4. Dense 레이어
    class Dense : public Layer {
    private:
        Matrix W_, b_, gW_, gb_;
        //가중치 행렬, 편향 벡터, W의 그래디언트, b의 그래티언트
    public:
        Dense(i32 in_dim, i32 out_dim, f32 init_scale = 0.01f)
            //행렬 크기 지정
            : W_(in_dim, out_dim), b_(1, out_dim), gW_(in_dim, out_dim), gb_(1, out_dim)
        {
            //Initializer를 사용한 가중치 초기화
            Initializer::Uniform(W_, init_scale, 123);
            b_.setZero();
        }

        //순전파: Y = X * W + b
        void Forward(const Matrix& X, Matrix& Y) override { 
            Y = X * W_; 
            Y.rowwise() += b_.row(0); //Y 모든 행에 b_를 더함**** for문을 대체
        }

        // 역전파: 그래디언트 계산
        void Backward(const Matrix& X, const Matrix& dY, Matrix& dX) override {
            // 가중치 그래디언트: gW = X^T * dY
            gW_ = X.transpose() * dY; //transpose() == 전치하는 함수
            // 편향 그래디언트: gb = sum_rows(dY)
            gb_ = dY.colwise().sum(); //행렬의 모든 열(column)을 각각 더하여 하나의 행 벡터를 만듬
            // 이전 레이어로 전달할 그래디언트: dX = dY * W^T
            dX = dY * W_.transpose();
        }

        //그래티언트 초기화
        void ZeroGrad() override { gW_.setZero(); gb_.setZero(); }

        //가중치 업데이트(경사하강법)
        void Step(float lr) override { W_ -= gW_ * lr; b_ -= gb_ * lr; }
    };

    // 5. ReLU 활성화 함수 레이어
    class ReLU : public Layer {
    private:
        // 역전파 계산을 위해 순전파 때의 입력값(X)을 저장해 둘 변수
        Matrix X_cache_;
    public:
        // 순전파: Y = max(0, X)
        void Forward(const Matrix& X, Matrix& Y) override {
            Y = X.array().max(0.f); //행렬(X)을 원소별 연산 모드로 전환하여 
            //각 원소에 독립적으로 계산. max는 배열 모드에서 원소가 양수면 그대로, 음수면 0
            X_cache_ = X;// 역전파에서 사용하기 위해 입력 X를 복사해 둠
        }

        // 역전파: 그래디언트 흐름 제어
        void Backward(const Matrix&, const Matrix& dY, Matrix& dX) override {
            // 순전파 때 입력(X_cache_)이 0보다 컸으면 1, 아니면 0인 마스크 생성
            // 이 마스크를 dY와 원소별로 곱하여 그래디언트 전달 여부 결정
            dX = (X_cache_.array() > 0.f).cast<f32>() * dY.array();
            //다시 배열 모드로 바꿔 각 원소가 0보다 큰지 비교 -> boolean으로
            //cast() -> boolean을 행렬 (1.0,0.0)으로 변환
            //* dY.array() 상위 레이어에서 전달된 그래디언트를 원소별로 곱함
        }
        //ReLU는 학습해야 할 가중치나 편향이 없으므로, ZeroGrad와 Step은 구현할 필요 없음
    };

    // 6. Loss 함수***(이해가 잘 안됨)
    class SoftmaxCrossEntropy {
    private:
        //순전파 시 계산된 Softmax 확률을 저장 (역전파 계산에 사용됨)
        Matrix probs_;
    public:
        // 순전파: Softmax 계산 후 Cross-Entropy Loss 반환
        float Forward(const Matrix& logits, const std::vector<int>& y) {
            const int N = logits.rows(); // 배치 크기

            // 1. 수치 안정성 확보: 각 행의 최댓값을 빼서 exp() 오버플로우 방지
            Matrix stable_logits = logits.colwise() - logits.rowwise().maxCoeff();

            // 2. 각 원소에 대해 지수 함수 적용
            Matrix exps = stable_logits.array().exp();

            // 3. 각 행의 합으로 나눠 Softmax 확률 계산 (결과는 probs_에 저장)
            probs_ = exps.array().colwise() / exps.rowwise().sum().array();

            // 4. Cross-Entropy Loss 계산
            float loss = 0.0f;
            for (int n = 0; n < N; ++n) {
                // log(0) 방지를 위해 아주 작은 값(epsilon)과 비교하여 최댓값 사용
                loss += -log(std::max(1e-12f, probs_(n, y[n]))); // y[n]은 정답 클래스 인덱스
            }
            // 배치 전체의 평균 Loss 반환
            return loss / static_cast<float>(N);
        }

        // 역전파: Softmax + CrossEntropy의 결합 그래디언트 계산
        void Backward(const std::vector<int>& y, Matrix& dLogits) {
            const int N = probs_.rows(); // 배치 크기

            // 1. 계산된 확률(probs_)을 그래디언트 행렬에 복사 (결합 미분 공식 시작)
            dLogits = probs_;

            // 2. 각 행(샘플)에서 정답 클래스(y[n])에 해당하는 열의 값에서 1을 빼줌
            //    (이것이 (예측 확률 - 정답) 공식을 구현하는 효율적인 방법)
            for (int n = 0; n < N; ++n) {
                dLogits(n, y[n]) -= 1.0f;
            }

            // 3. 배치 전체에 대한 평균 그래디언트를 만들기 위해 배치 크기로 나눠줌
            dLogits /= static_cast<float>(N);
        }
    };

    // 7. 학습 설정을 위한 구조체
    struct TrainConfig { int epochs = 1, batch = 1024, repeats = 1, warmup = 0; float lr = 0.01f; };
    struct TrainReport { float last_loss = -1.f; };

    // 8. Trainer 클래스 (학습 루프 포함)
    class Trainer {
    public:
        // 모델 학습을 수행하는 정적 함수
        //학습시킬 모델(model), 훈련 데이터(X, y), 학습 설정(cfg)
        static TrainReport Train(Sequential& model, const Matrix& X, const std::vector<int>& y, const TrainConfig& cfg) {
            const int N = X.rows(); // 전체 훈련 데이터 수
            std::vector<int> idx(N); // 데이터 인덱스를 저장할 벡터 (0, 1, ..., N-1)
            std::iota(idx.begin(), idx.end(), 0); // idx 벡터를 0부터 N-1까지 채움
            SoftmaxCrossEntropy loss_fn; // 손실 함수 객체 생성
            TrainReport report; // 학습 결과 리포트 객체 생성

            // [메모리 재사용] 배치 데이터를 담을 행렬과 벡터를 루프 시작 전에 선언
            Matrix X_batch;
            std::vector<int> y_batch;

            // --- 외부 반복 루프 (원본 코드의 repeats 구현) ---
            for (int r = 0; r < cfg.repeats; ++r) { // cfg.repeats (3) 만큼 반복
                std::cout << "--- Repeat " << r + 1 << "/" << cfg.repeats << " ---" << std::endl;

                // --- 내부 Epoch 루프 ---
                for (int e = 0; e < cfg.epochs; ++e) { // cfg.epochs (1) 만큼 반복
                    // 매 에포크(및 반복)마다 다른 시드로 데이터를 섞음 (재현성 유지)
                    std::mt19937 rng(e + r);
                    std::shuffle(idx.begin(), idx.end(), rng);

                    float epoch_loss = 0.f; // 현재 에포크의 누적 손실
                    int num_batches = 0; // 현재 에포크에서 처리한 배치 수

                    // --- 배치 처리 루프 ---
                    for (int i = 0; i < N; i += cfg.batch) { // 전체 데이터를 배치 크기만큼 건너뛰며 순회
                        // 현재 배치의 실제 크기 계산 (마지막 배치는 작을 수 있음)
                        int current_batch_size = std::min(cfg.batch, N - i);

                        // [메모리 재사용] X_batch, y_batch 크기 조정 (메모리 재할당 최소화)
                        X_batch.resize(current_batch_size, X.cols());
                        y_batch.resize(current_batch_size);

                        // 현재 배치에 해당하는 데이터를 X_batch, y_batch에 복사
                        for (int j = 0; j < current_batch_size; ++j) {
                            X_batch.row(j) = X.row(idx[i + j]); // Eigen 행 복사
                            y_batch[j] = y[idx[i + j]];
                        }

                        Matrix logits; // 모델 출력을 저장할 행렬

                        // --- 핵심 학습 단계 ---
                        // 1. 순전파: 현재 배치 데이터로 모델 예측 수행
                        model.Forward(X_batch, logits);
                        // 2. 손실 계산: 예측 결과와 정답을 비교하여 손실값 누적
                        epoch_loss += loss_fn.Forward(logits, y_batch);
                        num_batches++; // 처리한 배치 수 증가
                        // 3. 역전파 (그래디언트 계산 시작): 손실 함수부터 시작
                        Matrix dLogits, dX_dummy; // 그래디언트 저장용 임시 행렬
                        loss_fn.Backward(y_batch, dLogits);
                        // 4. 역전파 (모델 전체): 계산된 그래디언트를 모델에 흘려보냄
                        model.Backward(X_batch, dLogits, dX_dummy);
                        // 5. 가중치 업데이트: 계산된 그래디언트로 모델 파라미터 수정 (학습!)
                        model.Step(cfg.lr); // lr = 0.05
                        // 6. 그래디언트 초기화: 다음 배치를 위해 그래디언트 리셋
                        model.ZeroGrad();
                        // --- 핵심 학습 단계 끝 ---
                    } // --- 배치 루프 끝 ---

                    // 현재 에포크의 평균 손실 계산 및 보고
                    report.last_loss = epoch_loss / num_batches;
                    std::cout << "Epoch " << e + 1 << "/" << cfg.epochs << ", Loss: " << report.last_loss << std::endl;
                } // --- Epoch 루프 끝 ---
            } // --- 외부 반복 루프 끝 ---

            // 최종 학습 결과 반환
            return report;
        }
    };

    // 9. Timer 클래스
    class Timer {
    public:
        using clock = std::chrono::high_resolution_clock;
    private:
        clock::time_point t0_;
    public:
        void Tic() { t0_ = clock::now(); }
        double TocMs() const {
            auto t1 = clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0_).count();
        }
    };
}

// --- 유틸리티 함수들 (Refactored for Eigen) ---

static bool LoadCovertypeCSV(const std::string& path, vsnn::Matrix& X, std::vector<int>& y, int max_rows = -1, int stride = 1) {
    std::ifstream fin(path);
    if (!fin.is_open()) return false;
    std::string line; int row = 0; int kept = 0;
    std::vector<std::array<float, 54>> feats; feats.reserve(10000);
    std::vector<int> labels; labels.reserve(10000);
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        if ((row++ % stride) != 0) continue;
        std::stringstream ss(line);
        std::string tok; std::array<float, 54> f{};
        int col = 0; bool ok = true; float v = 0.f;
        for (; col < 54; ++col) {
            if (!std::getline(ss, tok, ',')) { ok = false; break; }
            try { v = std::stof(tok); }
            catch (...) { ok = false; break; }
            f[col] = v;
        }
        if (!ok) continue;
        if (!std::getline(ss, tok, ',')) continue;
        int lab = 0; try { lab = std::stoi(tok); }
        catch (...) { continue; }
        if (lab < 1 || lab > 7) continue;
        labels.push_back(lab - 1);
        feats.push_back(f);
        ++kept;
        if (max_rows > 0 && kept >= max_rows) break;
    }
    const int N = static_cast<int>(feats.size());
    if (N == 0) return false;
    X.resize(N, 54);
    y = labels;
    for (int n = 0; n < N; ++n) for (int d = 0; d < 54; ++d) X(n, d) = feats[n][d];
    return true;
}

//로드된 데이터 10가지 특성에 대한 표준화
static void StandardizeCovertype(vsnn::Matrix& X) {
    const int N = X.rows();
    const int cont = 10;
    for (int d = 0; d < cont; ++d) {
        float mean = X.col(d).mean();
        float stddev = std::sqrt((X.col(d).array() - mean).square().sum() / N);
        if (stddev == 0.f) stddev = 1.f;

        //X.col(d) = (X.col(d).array() - mean) / stddev;
        X.col(d).array() -= mean;   // 1. d번째 열의 모든 원소에서 평균을 뺍니다.***
        X.col(d).array() /= stddev; // 2. d번째 열의 모든 원소를 표준편차로 나눕니다.***
    }
}

//훈련용과 테스트용으로 분리
static void GatherRows(const vsnn::Matrix& X, const std::vector<int>& y, const std::vector<int>& idx, vsnn::Matrix& Xo, std::vector<int>& yo) {
    const int N = (int)idx.size(), D = X.cols();
    Xo.resize(N, D);
    yo.resize(N);
    for (int i = 0; i < N; ++i) {
        Xo.row(i) = X.row(idx[i]);
        yo[i] = y[idx[i]];
    }
}

//신경망 모델의 최족 출력을 박아 가장 높은 점수를 가진 클래스의 인덱스를 찾아 반환
//모델이 - 가장 가능성이 높다고 예측한 클래스가 무엇인지 알려주는 클래스
static int ArgMax(const vsnn::Matrix& vec) {
    Eigen::Index max_idx; // Eigen::Index 타입 사용
    vec.row(0).maxCoeff(&max_idx);
    //vec.row(0) 입력 행렬 vec의 첫번째 행을 가져옴
    //.maxCoeff(&max_idx) 해당 백터에서 가장 큰 원소를 찾고 그 원소의 인덱스 저장
    return static_cast<int>(max_idx);
}


// --- main 함수 ---
int main() {
    using namespace vsnn;
    using namespace std;

    // 1. 데이터 로드 및 전처리
    Matrix X; vector<int> y;
    if (!LoadCovertypeCSV("covtype.data", X, y, 120000, 2)) {
        cerr << "[ERROR] covtype.data 로드 실패." << endl; return 1;
    }
    StandardizeCovertype(X);

    // 2. 훈련/테스트 데이터 분할
    const int N = X.rows();
    vector<int> idx(N);
    iota(idx.begin(), idx.end(), 0);
    mt19937 rng(0);
    shuffle(idx.begin(), idx.end(), rng);

    const int Ntrain = (int)(N * 0.9);
    vector<int> idx_tr(idx.begin(), idx.begin() + Ntrain), idx_te(idx.begin() + Ntrain, idx.end());
    Matrix Xtr, Xte; vector<int> ytr, yte;
    GatherRows(X, y, idx_tr, Xtr, ytr);
    GatherRows(X, y, idx_te, Xte, yte);
    cout << "[Dataset] rows=" << N << " (train=" << Xtr.rows() << ", test=" << Xte.rows() << ") D=54 C=7" << endl;

    // 3. 모델 생성 (Sequential에 Layer 추가)
    Sequential model;
    model.Add<Dense>(54, 256);
    model.Add<ReLU>();
    model.Add<Dense>(256, 7);

    // 4. 학습 전 추론 (초기 상태 확인)
    cout << "\n[Inference-only before training]" << endl;
    for (int n = 0; n < min(5, (int)yte.size()); ++n) {
        Matrix logits;
        model.Forward(Xte.row(n), logits);
        int pred = ArgMax(logits);
        SoftmaxCrossEntropy CE1; vector<int> y1(1, yte[n]); //단일 샘플용 Loss 객체
        float loss1 = CE1.Forward(logits, y1);
        cout << fixed << setprecision(4) << "\n sample index:" << n
            << " pred=" << pred << " answer=" << yte[n] << " loss=" << loss1;
    }

    // 5. 학습 실행
    vsnn::Timer TotalTimer;
    TotalTimer.Tic(); //타이머 시작
    TrainConfig cfg;
    cfg.epochs = 1; cfg.batch = 1024; cfg.lr = 5e-2f;
    cfg.warmup = 1;
    cfg.repeats = 3;

    auto report = Trainer::Train(model, Xtr, ytr, cfg); //****** 학습 실행 ******//
    double total_ms = TotalTimer.TocMs();

    // 6. 학습 결과 보고
    cout << "\n\n[Training report]" << endl;
    cout << " total training time (ms): " << total_ms << endl;
    cout << " final loss : " << report.last_loss << endl;

    // 7. 학습 후 추론 (결과 비교)
    cout << "\n[Inference-only after training]" << endl;
    for (int n = 0; n < min(5, (int)yte.size()); ++n) {
        Matrix logits;
        model.Forward(Xte.row(n), logits);
        int pred = ArgMax(logits);
        SoftmaxCrossEntropy CE1; vector<int> y1(1, yte[n]);
        float loss1 = CE1.Forward(logits, y1);
        cout << fixed << setprecision(4) << "\n sample index:" << n
            << " pred=" << pred << " answer=" << yte[n] << " loss=" << loss1;
    }

    // 8. 테스트 정확도 계산 및 출력
    int correct = 0; //맞춘 개수
    for (int n = 0; n < Xte.rows(); ++n) {
        Matrix logits;
        // 각 테스트 샘플로 순전파 수행
        model.Forward(Xte.row(n), logits);
        // 모델의 예측(ArgMax)과 실제 정답(yte[n])이 같으면 카운트 증가
        if (ArgMax(logits) == yte[n]) ++correct;
    }

    // 정확도 계산: (맞춘 개수 / 전체 테스트 데이터 개수)
    double acc = (double)correct / std::max(1, (int)Xte.rows());
    cout << "\n\n[Test accuracy] " << fixed << setprecision(4) << acc << endl;

    return 0;
}
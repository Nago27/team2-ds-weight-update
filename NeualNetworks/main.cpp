// =============================
// src/main.cpp
// =============================
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <array>

// --- 프로젝트 헤더 ---
#include "Matrix.hpp"
#include "Dense.hpp"
#include "Activations.hpp"
#include "Loss.hpp"
#include "Sequential.hpp"
#include "Timer.hpp"
#include "Trainer.hpp"

using namespace vsnn;
using namespace std;

// ❗ 원본 파일에 있던 함수들을 다시 포함해야 합니다.
static bool LoadCovertypeCSV(const std::string& path, Matrix& X, vector<int>& y,
    int max_rows = -1, int stride = 1) {
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

    const int N_data = static_cast<int>(feats.size());
    if (N_data == 0) return false;
    X.resize(N_data, 54); y = labels;
    for (int n = 0; n < N_data; ++n) for (int d = 0; d < 54; ++d) X(n, d) = feats[n][d];

    return true;
}

static void StandardizeCovertype(Matrix& X) {
    const int N_data = X.rows();
    const int cont = 10;
    vector<float> mean(cont, 0.f), stdv(cont, 0.f);
    for (int d = 0; d < cont; ++d) {
        for (int n = 0; n < N_data; ++n) mean[d] += X(n, d);
        mean[d] /= std::max(1, N_data);
        for (int n = 0; n < N_data; ++n) { float z = X(n, d) - mean[d]; stdv[d] += z * z; }
        stdv[d] = std::sqrt(stdv[d] / std::max(1, N_data));
        if (stdv[d] == 0.f) stdv[d] = 1.f;
        for (int n = 0; n < N_data; ++n) X(n, d) = (X(n, d) - mean[d]) / stdv[d];
    }
}

static void GatherRows(const Matrix& X, const vector<int>& y, const vector<int>& idx, Matrix& Xo, vector<int>& yo) {
    const int N_data = (int)idx.size(), D = X.cols();
    Xo.resize(N_data, D);
    yo.resize(N_data);
    for (int i = 0; i < N_data; ++i) {
        Xo.row(i) = X.row(idx[i]);
        yo[i] = y[idx[i]];
    }
}

static int ArgMaxRow0(const Matrix& logits) {
    Matrix::Index max_index;
    logits.row(0).maxCoeff(&max_index);
    return static_cast<int>(max_index);
}

int main() {
    // ---------------------------------------------------------
    // 0) 데이터 준비 (❗ 생략되었던 부분을 다시 복원)
    // ---------------------------------------------------------
    Matrix X; vector<int> y;
    const string path = "covtype.data";
    const int max_rows = 120000;
    const int stride = 2;

    if (!LoadCovertypeCSV(path, X, y, max_rows, stride)) {
        cerr << "[ERROR] " << path << " 로드 실패. 경로/포맷을 확인하세요." << endl;
        return 1;
    }
    StandardizeCovertype(X);

    const int N = X.rows(); // ❗ N 변수 선언
    vector<int> idx(N);
    iota(idx.begin(), idx.end(), 0);
    mt19937 rng(0);
    shuffle(idx.begin(), idx.end(), rng);

    const int Ntrain = (int)(N * 0.9);
    vector<int> idx_tr(idx.begin(), idx.begin() + Ntrain), idx_te(idx.begin() + Ntrain, idx.end());
    Matrix Xtr, Xte; // ❗ Xtr, Xte 변수 선언
    vector<int> ytr, yte; // ❗ ytr, yte 변수 선언
    GatherRows(X, y, idx_tr, Xtr, ytr);
    GatherRows(X, y, idx_te, Xte, yte);

    cout << "[Dataset] rows=" << N << " (train=" << Xtr.rows() << ", test=" << Xte.rows() << ") D=54 C=7" << endl;

    // ---------------------------------------------------------
    // 1) 모델 구성
    // ---------------------------------------------------------
    Sequential model; model.Add<Dense>(54, 256); model.Add<ReLU>(); model.Add<Dense>(256, 7);

    // ---------------------------------------------------------
    // 2) 트레이닝 전/후 피드포워드 확인
    // ---------------------------------------------------------
    auto run_inference = [&](const string& title) {
        cout << "\n" << endl;
        cout << title << endl;
        cout << "five examples" << endl;
        // 💡 수정: min -> std::min
        for (int n = 0; n < std::min(5, (int)yte.size()); ++n) {
            Matrix X1 = Xte.row(n);
            Matrix logits;
            model.Forward(X1, logits);
            int pred = ArgMaxRow0(logits);
            SoftmaxCrossEntropy CE1;
            vector<int> y1(1, yte[n]);
            float loss1 = CE1.Forward(logits, y1);
            cout << "\n" << endl;
            cout << fixed << setprecision(4)
                << " sample index:" << n
                << " pred=" << pred
                << " answer=" << yte[n]
                << " loss=" << loss1 << "";
        }
        };

    run_inference("[Inference-only before training]");

    // ---------------------------------------------------------
    // 3) 학습 실행 (❗ 생략되었던 부분을 다시 복원)
    // ---------------------------------------------------------
    Timer TotalTimer;
    TotalTimer.Tic();
    TrainConfig cfg;
    cfg.epochs = 1; cfg.batch = 1024; cfg.lr = 5e-2f; cfg.warmup = 1; cfg.repeats = 3;
    auto report = Trainer::Train<TrainUpdater>(model, Xtr, ytr, cfg); // X,y 대신 Xtr, ytr 사용
    double total_ms = TotalTimer.TocMs();

    cout << "\n" << endl;
    cout << "[Training report]" << endl;
    cout << " total training time (ms): " << total_ms << "\n";
    cout << " final loss : " << report.last_loss << "";

    run_inference("[Inference-only after training]");

    // ---------------------------------------------------------
    // 5) 테스팅 후 정확도 확인
    // ---------------------------------------------------------
    int correct = 0;
    Matrix logits;
    for (int n = 0; n < Xte.rows(); ++n) {
        Matrix X1 = Xte.row(n);
        model.Forward(X1, logits);
        if (ArgMaxRow0(logits) == yte[n]) ++correct;
    }
    // 💡 수정: max -> std::max
    double acc = (double)correct / std::max(1, (int)Xte.rows());
    cout << "\n" << endl;
    cout << "[Test accuracy] " << fixed << setprecision(4) << acc << endl;

    return 0;
}
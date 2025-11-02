<<<<<<< HEAD
// =============================
// src/main.cpp
// =============================
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

=======
ï»?/ =============================
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

// --- ?„ë¡œ?íŠ¸ ?¤ë” ---
>>>>>>> 2793eccbf0cde4fe88631a61da30e526f24455d2
#include "Matrix.hpp"
#include "Dense.hpp"
#include "Activations.hpp"
#include "Loss.hpp"
#include "Sequential.hpp"
<<<<<<< HEAD
#include "Perceptron.hpp"
#include "Timer.hpp"
#include "Trainer.hpp"
#include "Ops.hpp"

=======
#include "Timer.hpp"
#include "Trainer.hpp"
>>>>>>> 2793eccbf0cde4fe88631a61da30e526f24455d2

using namespace vsnn;
using namespace std;

<<<<<<< HEAD
static bool LoadCovertypeCSV(const std::string& path, Matrix& X, vector<int>& y, 
	int max_rows = -1, int stride = 1) {
	std::ifstream fin(path);
	if (!fin.is_open()) return false;
	std::string line; int row = 0; int kept = 0;


	std::vector<std::array<float, 54>> feats; feats.reserve(10000);
	std::vector<int> labels; labels.reserve(10000);

	while (std::getline(fin, line)) {
		if (line.empty()) continue;
		if ((row++ % stride) != 0) continue; // ¼­ºê»ùÇÃ¸µ
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
		if (!std::getline(ss, tok, ',')) continue; // class
		int lab = 0; try { lab = std::stoi(tok); }
		catch (...) { continue; }
		if (lab < 1 || lab > 7) continue; // 1..7
		labels.push_back(lab - 1); // 0..6
		feats.push_back(f);
		++kept;
		if (max_rows > 0 && kept >= max_rows) break;
	}

	const int N = static_cast<int>(feats.size());
	if (N == 0) return false;
	X.Reset(N, 54); y = labels;
	for (int n = 0; n < N; ++n) for (int d = 0; d < 54; ++d) X(n, d) = feats[n][d];

	return true;
}

static void StandardizeCovertype(Matrix& X) {
	// ¿¬¼ÓÇü 10°³¸¸ Ç¥ÁØÈ­, one-hot 44°³´Â ±×´ë·Î
	const int N = X.Rows();
	const int D = X.Cols();
	const int cont = 10; // 0..9
	vector<float> mean(cont, 0.f), stdv(cont, 0.f);
	for (int d = 0; d < cont; ++d) {
		for (int n = 0; n < N; ++n) mean[d] += X(n, d);
		mean[d] /= std::max(1, N);
		for (int n = 0; n < N; ++n) { float z = X(n, d) - mean[d]; stdv[d] += z * z; }
		stdv[d] = std::sqrt(stdv[d] / std::max(1, N));
		if (stdv[d] == 0.f) stdv[d] = 1.f;
		for (int n = 0; n < N; ++n) X(n, d) = (X(n, d) - mean[d]) / stdv[d];
	}
	(void)D; // ³ª¸ÓÁö 44°³´Â 0/1 ±×´ë·Î À¯Áö
}


static void GatherRows(const Matrix& X, const vector<int>& y, const vector<int>& idx, Matrix& Xo, vector<int>& yo) {
	const int N = (int)idx.size(), D = X.Cols();
	Xo.Reset(N, D); yo.resize(N);
	for (int i = 0; i < N; ++i) { int n = idx[i]; for (int d = 0; d < D; ++d) Xo(i, d) = X(n, d); yo[i] = y[n]; }
}


static int ArgMaxRow0(const Matrix& logits) {
	int C = logits.Cols(); int bi = 0; float bv = logits(0, 0);
	for (int j = 1; j < C; ++j) { if (logits(0, j) > bv) { bv = logits(0, j); bi = j; } }
	return bi;
}

int main() {
	// ---------------------------------------------------------
	// 0) µ¥ÀÌÅÍ ÁØºñ
	// ---------------------------------------------------------
	Matrix X; vector<int> y;
	const string path = "covtype.data"; // UCI ¿øº» ÆÄÀÏ¸í
	const int max_rows = 120000; // ÀüÃ¼(581k) Áß »óÇÑ. ÀüÃ¼ ¾²·Á¸é -1·Î.
	const int stride = 2; // 2·Î ÇÏ¸é Àı¹İ »ùÇÃ »ç¿ë. ´õ ÁÙÀÌ·Á¸é 4,8...


	if (!LoadCovertypeCSV(path, X, y, max_rows, stride)) {
		cerr << "[ERROR] " << path << " ·Îµå ½ÇÆĞ. °æ·Î/Æ÷¸ËÀ» È®ÀÎÇÏ¼¼¿ä." << endl;
		return 1;
	}
	StandardizeCovertype(X);

	const int N = X.Rows(); vector<int> idx(N); iota(idx.begin(), idx.end(), 0);
	mt19937 rng(0); shuffle(idx.begin(), idx.end(), rng);
	const int Ntrain = (int)(N * 0.9);
	vector<int> idx_tr(idx.begin(), idx.begin() + Ntrain), idx_te(idx.begin() + Ntrain, idx.end());
	Matrix Xtr, Xte; vector<int> ytr, yte; GatherRows(X, y, idx_tr, Xtr, ytr); GatherRows(X, y, idx_te, Xte, yte);

	cout << "[Dataset] rows=" << N << " (train=" << Xtr.Rows() << ", test=" << Xte.Rows() << ") D=54 C=7" << endl;


	// ---------------------------------------------------------
	// 1) ¸ğµ¨ ±¸¼º
	// ---------------------------------------------------------
	Sequential model; model.Add<Dense>(54, 256); model.Add<ReLU>(); model.Add<Dense>(256, 7); // ÀÌ ºÎºĞÀº Àı´ë °ÇµéÁö ¸¶¼¼¿ä!!


	// ---------------------------------------------------------
	// 2) (¿äÃ»»çÇ×) Æ®·¹ÀÌ´× Àü¿¡ ÇÇµåÆ÷¿öµå¸¸ µ¹·Á¼­ Ãâ·Â È®ÀÎ
	// ---------------------------------------------------------
	cout << "[Inference-only before training]" << endl;
	cout << "five examples" << endl;
	for (int n = 0; n < min(5, (int)yte.size()); ++n) {
		Matrix X1(1, Xte.Cols()); for (int d = 0; d < Xte.Cols(); ++d) X1(0, d) = Xte(n, d);
		Matrix logits; model.Forward(X1, logits);
		int pred = ArgMaxRow0(logits);
		SoftmaxCrossEntropy CE1; vector<int> y1(1, yte[n]);
		float loss1 = CE1.Forward(logits, y1);
		cout << "\n" << endl;
		cout << fixed << setprecision(4)
			<< " sample index:" << n
			<< " pred=" << pred
			<< " answer=" << yte[n]
			<< " loss=" << loss1 << "";
	}

	// ---------------------------------------------------------
	// 3) ÇĞ½À ½ÇÇà (¾÷µ¥ÀÌÆ® ½Ã°£ º°µµ ÃøÁ¤: Trainer -> StudentUpdater::Update)
	// ---------------------------------------------------------
	Timer TotalTimer;
	TotalTimer.Tic();

	TrainConfig cfg; cfg.epochs = 1; cfg.batch = 1024; cfg.lr = 5e-2f; cfg.warmup = 1; cfg.repeats = 3; // ÀÌ ºÎºĞÀº Àı´ë °ÇµéÁö ¸¶¼¼¿ä!!
	auto report = Trainer::Train<TrainUpdater>(model, X, y, cfg);
	double total_ms = TotalTimer.TocMs();

	cout << "\n" << endl;
	cout << "[Training report]" << endl;
	cout << " total training time (ms): " << total_ms << "\n";
	cout << " final loss : " << report.last_loss << "";

	// ---------------------------------------------------------
	// 4) Æ®·¹ÀÌ´× ÈÄ ÇÇµåÆ÷¿öµå °á°ú ÀçÈ®ÀÎ
	// ---------------------------------------------------------
	cout << "\n" << endl;
	cout << "[Inference - only after training]" << endl;
	cout << "five examples" << endl;
	for (int n = 0; n < min(5, (int)yte.size()); ++n) {
		Matrix X1(1, Xte.Cols()); for (int d = 0; d < Xte.Cols(); ++d) X1(0, d) = Xte(n, d);
		Matrix logits; model.Forward(X1, logits);
		int pred = ArgMaxRow0(logits);
		SoftmaxCrossEntropy CE1; vector<int> y1(1, yte[n]);
		float loss1 = CE1.Forward(logits, y1);
		cout << "\n" << endl;
		cout << fixed << setprecision(4)
			<< " sample index:" << n
			<< " pred=" << pred
			<< " answer=" << yte[n]
			<< " loss=" << loss1 << "";
	}

	// ---------------------------------------------------------
	// 5) Å×½ºÆÃ ÈÄ Á¤È®µµ È®ÀÎ
	// ---------------------------------------------------------
	int correct = 0; Matrix logits;
	for (int n = 0; n < Xte.Rows(); ++n) {
		Matrix X1(1, Xte.Cols()); for (int d = 0; d < Xte.Cols(); ++d) X1(0, d) = Xte(n, d);
		model.Forward(X1, logits);
		if (ArgMaxRow0(logits) == yte[n]) ++correct;
	}
	double acc = (double)correct / max(1, Xte.Rows());
	cout << "\n" << endl;
	cout << "[Test accuracy] " << fixed << setprecision(4) << acc << endl;

	return 0;

=======
// ???ë³¸ ?Œì¼???ˆë˜ ?¨ìˆ˜?¤ì„ ?¤ì‹œ ?¬í•¨?´ì•¼ ?©ë‹ˆ??
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
    // 0) ?°ì´??ì¤€ë¹?(???ëµ?˜ì—ˆ??ë¶€ë¶„ì„ ?¤ì‹œ ë³µì›)
    // ---------------------------------------------------------
    Matrix X; vector<int> y;
    const string path = "covtype.data";
    const int max_rows = 120000;
    const int stride = 2;

    if (!LoadCovertypeCSV(path, X, y, max_rows, stride)) {
        cerr << "[ERROR] " << path << " ë¡œë“œ ?¤íŒ¨. ê²½ë¡œ/?¬ë§·???•ì¸?˜ì„¸??" << endl;
        return 1;
    }
    StandardizeCovertype(X);

    const int N = X.rows(); // ??N ë³€??? ì–¸
    vector<int> idx(N);
    iota(idx.begin(), idx.end(), 0);
    mt19937 rng(0);
    shuffle(idx.begin(), idx.end(), rng);

    const int Ntrain = (int)(N * 0.9);
    vector<int> idx_tr(idx.begin(), idx.begin() + Ntrain), idx_te(idx.begin() + Ntrain, idx.end());
    Matrix Xtr, Xte; // ??Xtr, Xte ë³€??? ì–¸
    vector<int> ytr, yte; // ??ytr, yte ë³€??? ì–¸
    GatherRows(X, y, idx_tr, Xtr, ytr);
    GatherRows(X, y, idx_te, Xte, yte);

    cout << "[Dataset] rows=" << N << " (train=" << Xtr.rows() << ", test=" << Xte.rows() << ") D=54 C=7" << endl;

    // ---------------------------------------------------------
    // 1) ëª¨ë¸ êµ¬ì„±
    // ---------------------------------------------------------
    Sequential model; model.Add<Dense>(54, 256); model.Add<ReLU>(); model.Add<Dense>(256, 7);

    // ---------------------------------------------------------
    // 2) ?¸ë ˆ?´ë‹ ?????¼ë“œ?¬ì›Œ???•ì¸
    // ---------------------------------------------------------
    auto run_inference = [&](const string& title) {
        cout << "\n" << endl;
        cout << title << endl;
        cout << "five examples" << endl;
        // ?’¡ ?˜ì •: min -> std::min
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
    // 3) ?™ìŠµ ?¤í–‰ (???ëµ?˜ì—ˆ??ë¶€ë¶„ì„ ?¤ì‹œ ë³µì›)
    // ---------------------------------------------------------
    Timer TotalTimer;
    TotalTimer.Tic();
    TrainConfig cfg;
    cfg.epochs = 1; cfg.batch = 1024; cfg.lr = 5e-2f; cfg.warmup = 1; cfg.repeats = 3;
    auto report = Trainer::Train<TrainUpdater>(model, Xtr, ytr, cfg); // X,y ?€??Xtr, ytr ?¬ìš©
    double total_ms = TotalTimer.TocMs();

    cout << "\n" << endl;
    cout << "[Training report]" << endl;
    cout << " total training time (ms): " << total_ms << "\n";
    cout << " final loss : " << report.last_loss << "";

    run_inference("[Inference-only after training]");

    // ---------------------------------------------------------
    // 5) ?ŒìŠ¤?????•í™•???•ì¸
    // ---------------------------------------------------------
    int correct = 0;
    Matrix logits;
    for (int n = 0; n < Xte.rows(); ++n) {
        Matrix X1 = Xte.row(n);
        model.Forward(X1, logits);
        if (ArgMaxRow0(logits) == yte[n]) ++correct;
    }
    // ?’¡ ?˜ì •: max -> std::max
    double acc = (double)correct / std::max(1, (int)Xte.rows());
    cout << "\n" << endl;
    cout << "[Test accuracy] " << fixed << setprecision(4) << acc << endl;

    return 0;
>>>>>>> 2793eccbf0cde4fe88631a61da30e526f24455d2
}
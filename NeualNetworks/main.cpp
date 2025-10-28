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

#include "Matrix.hpp"
#include "Dense.hpp"
#include "Activations.hpp"
#include "Loss.hpp"
#include "Sequential.hpp"
#include "Perceptron.hpp"
#include "Timer.hpp"
#include "Trainer.hpp"
#include "Ops.hpp"


using namespace vsnn;
using namespace std;

//**머신러닝에 필요한 데이터 셋을 생성하는 역할의 함수. covtype.data에서 읽어옴
static bool LoadCovertypeCSV(const std::string& path, Matrix& X, vector<int>& y, 
	int max_rows = -1, int stride = 1) {
	std::ifstream fin(path); //**지정된 path의 CSV파일을 연다
	if (!fin.is_open()) return false; //**정상적으로 열리지 않으면 false
	std::string line; int row = 0; int kept = 0;

	//**데이터를 임시로 저장할 vector 두 개 생성
	std::vector<std::array<float, 54>> feats; feats.reserve(10000);//**각 행의 54개 특성 저장 백터
	std::vector<int> labels; labels.reserve(10000);//**각 데이터에 부여된 정답(레이블)
	//**reserve 미리 메모리 공간을 할당하여 데이터가 추가될 때마다 재할당하는 비효율 줄임

	while (std::getline(fin, line)) {
		if (line.empty()) continue;
		if ((row++ % stride) != 0) continue; // 서브샘플링
		//**서브샘플링-> stride 매개변수를 사용해 모든 행을 읽지 않고 지정된 간격으로
		//**행을 건너뛰어 일부 데이터만 사용. 1이면 모든 행, 2면 한 줄 읽고 한줄 건너뛰기
		std::stringstream ss(line); //** 한 줄의 문자열을 쉼표를 기준으로 자름
		std::string tok; std::array<float, 54> f{};
		int col = 0; bool ok = true; float v = 0.f;
		for (; col < 54; ++col) { //**54개 특성값을 읽어 feats 벡터에 저장
			if (!std::getline(ss, tok, ',')) { ok = false; break; }
			try { v = std::stof(tok); }
			catch (...) { ok = false; break; }
			f[col] = v;
		}
		if (!ok) continue;
		if (!std::getline(ss, tok, ',')) continue; // 55번째 값 -> 레이블을 읽음
		int lab = 0; try { lab = std::stoi(tok); }
		catch (...) { continue; }
		if (lab < 1 || lab > 7) continue; // 1..7
		labels.push_back(lab - 1); // 0..6  **레이블 추출
		feats.push_back(f); 
		++kept;
		if (max_rows > 0 && kept >= max_rows) break; //**읽어들일 행의 개수 제한
	}

	//**모든 행을 읽은 후, 임시로 저장했던 feats와 labels의 데이터를 최종적인 출력
	//**데이터 구조인 Matrix& X와 vector<int>& y로 옮김
	const int N = static_cast<int>(feats.size()); //**N행->feats의 크기
	if (N == 0) return false;
	X.Reset(N, 54); y = labels; //Matrix& X는 N행과 54열로 구성된 2차원 행렬로 구성
	//**레이블은 vector<int>& y에 저장
	for (int n = 0; n < N; ++n) for (int d = 0; d < 54; ++d) X(n, d) = feats[n][d];

	return true;
}

static void StandardizeCovertype(Matrix& X) {
	// 연속형 10개만 표준화, one-hot 44개는 그대로
	//**앞 10개만 연속적인 숫자이기에 큰 값을 가진 특성이 학습에 불균형적 영향을 미침
	const int N = X.Rows();
	const int D = X.Cols();
	const int cont = 10; // 0..9 **맨 앞 10개 특성
	vector<float> mean(cont, 0.f), stdv(cont, 0.f);
	for (int d = 0; d < cont; ++d) {
		for (int n = 0; n < N; ++n) mean[d] += X(n, d);
		mean[d] /= std::max(1, N); //**10개 특성의 평균값
		for (int n = 0; n < N; ++n) { float z = X(n, d) - mean[d]; stdv[d] += z * z; }
		stdv[d] = std::sqrt(stdv[d] / std::max(1, N)); //**10개 특성의 표준편차
		if (stdv[d] == 0.f) stdv[d] = 1.f;
		for (int n = 0; n < N; ++n) X(n, d) = (X(n, d) - mean[d]) / stdv[d]; //**표준화 적용
	}
	(void)D; // 나머지 44개는 0/1 그대로 유지
}

//** 원본 데이터에서 주어진 인덱스(idx)에 해단하는 행들만 골라 새 데이터셋을 만드는 함수
//** 학습용과 테스트용을 분리하는데 사용되는 것으로 보임
static void GatherRows(const Matrix& X, const vector<int>& y, const vector<int>& idx, Matrix& Xo, vector<int>& yo) {
	const int N = (int)idx.size(), D = X.Cols();
	Xo.Reset(N, D); yo.resize(N);
	for (int i = 0; i < N; ++i) { int n = idx[i]; for (int d = 0; d < D; ++d) Xo(i, d) = X(n, d); yo[i] = y[n]; }
}

//** 신경망의 출력값(logits)중에서 가장 큰 값을 가진 인덱스를 찾는 역할
//** logits -> 모델이 출력한 7개 클래스에 대한 점수를 담고 있는 1x7 크기의 행렬
static int ArgMaxRow0(const Matrix& logits) {
	int C = logits.Cols(); int bi = 0; float bv = logits(0, 0);
	for (int j = 1; j < C; ++j) { if (logits(0, j) > bv) { bv = logits(0, j); bi = j; } }
	return bi; //**가장 큰 값을 가진 인덱스 bi 를 반환
}

int main() {
	// ---------------------------------------------------------
	// 0) 데이터 준비
	// ---------------------------------------------------------
	Matrix X; vector<int> y;
	const string path = "covtype.data"; // UCI 원본 파일명
	const int max_rows = 120000; // 전체(581k) 중 상한. 전체 쓰려면 -1로.
	const int stride = 2; // 2로 하면 절반 샘플 사용. 더 줄이려면 4,8...

	//**데이터 읽고 각 매개변수에 저장
	if (!LoadCovertypeCSV(path, X, y, max_rows, stride)) {
		cerr << "[ERROR] " << path << " 로드 실패. 경로/포맷을 확인하세요." << endl;
		return 1;
	}
	StandardizeCovertype(X); //**연속형 특성 10개를 표준화

	const int N = X.Rows(); vector<int> idx(N); iota(idx.begin(), idx.end(), 0);
	mt19937 rng(0); shuffle(idx.begin(), idx.end(), rng); 
	//**shuffle 데이터 섞기. 학습 편향을 방지
	const int Ntrain = (int)(N * 0.9); //**전체 데이터 90%는 학습용, 10% 테스트용
	vector<int> idx_tr(idx.begin(), idx.begin() + Ntrain), idx_te(idx.begin() + Ntrain, idx.end());
	Matrix Xtr, Xte; vector<int> ytr, yte; GatherRows(X, y, idx_tr, Xtr, ytr); GatherRows(X, y, idx_te, Xte, yte);
	//** xtr,ytr은 training / xte,yte는 testing
	cout << "[Dataset] rows=" << N << " (train=" << Xtr.Rows() << ", test=" << Xte.Rows() << ") D=54 C=7" << endl;


	// ---------------------------------------------------------
	// 1) 모델 구성
	// ---------------------------------------------------------
	Sequential model; model.Add<Dense>(54, 256); model.Add<ReLU>(); model.Add<Dense>(256, 7); // 이 부분은 절대 건들지 마세요!!


	// ---------------------------------------------------------
	// 2) (요청사항) 트레이닝 전에 피드포워드만 돌려서 출력 확인
	// ---------------------------------------------------------
	//**학습을 하지 않고 가중치가 무작위로 초기화된 상태의 모델이 어떤 예측을 하는가?
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
	// 3) 학습 실행 (업데이트 시간 별도 측정: Trainer -> StudentUpdater::Update)
	// ---------------------------------------------------------
	Timer TotalTimer;
	TotalTimer.Tic();

	//**학습 실행
	TrainConfig cfg; cfg.epochs = 1; cfg.batch = 1024; cfg.lr = 5e-2f; cfg.warmup = 1; cfg.repeats = 3; // 이 부분은 절대 건들지 마세요!!
	auto report = Trainer::Train<TrainUpdater>(model, X, y, cfg);
	double total_ms = TotalTimer.TocMs(); //**전체 학습시간 측정

	cout << "\n" << endl;
	cout << "[Training report]" << endl;
	cout << " total training time (ms): " << total_ms << "\n";
	cout << " final loss : " << report.last_loss << "";

	// ---------------------------------------------------------
	// 4) 트레이닝 후 피드포워드 결과 재확인
	// ---------------------------------------------------------
	//**학습 이후 실행 결과 확인
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
	// 5) 테스팅 후 정확도 확인
	// ---------------------------------------------------------
	int correct = 0; Matrix logits;
	for (int n = 0; n < Xte.Rows(); ++n) {
		Matrix X1(1, Xte.Cols()); for (int d = 0; d < Xte.Cols(); ++d) X1(0, d) = Xte(n, d);
		model.Forward(X1, logits);
		//**모델의 예측값과 실제 정답이 같으면 correct 1씩 증가
		if (ArgMaxRow0(logits) == yte[n]) ++correct; 
	}
	//**정확도(acc) = 정답을 맞힌 개수 / 전체 테스트 데이터 개수
	double acc = (double)correct / max(1, Xte.Rows());
	cout << "\n" << endl;
	cout << "[Test accuracy] " << fixed << setprecision(4) << acc << endl;

	return 0;

}
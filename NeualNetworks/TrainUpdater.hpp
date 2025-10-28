// =============================
// include/vsnn/TrainUpdater.hpp (올바른 버전)
// =============================
#pragma once
#include "Sequential.hpp"
#include "Dense.hpp" // Dense 레이어의 멤버에 접근해야 하므로 포함

namespace vsnn {
    // 경사 하강법을 이용해 모델 파라미터를 업데이트하는 클래스입니다.
    class TrainUpdater {
    public:
        // 모델의 모든 레이어를 순회하며 가중치를 업데이트합니다.
        static void Update(Sequential& model, float lr) {
            for (size_t i = 0; i < model.NumLayers(); ++i) {
                // 현재 레이어가 Dense 타입인지 확인합니다.
                if (auto* layer = dynamic_cast<Dense*>(model.LayerAt(i))) {

                    // Eigen의 벡터화 연산을 사용하여 가중치와 편향을 한 번에 업데이트합니다.
                    // 이는 수동 for 루프보다 훨씬 빠르고 효율적입니다.
                    // 공식: W_new = W_old - learning_rate * gW
                    layer->WRef() -= layer->gWRef() * lr;
                    layer->bRef() -= layer->gbRef() * lr;
                }
            }
        }
    };
}
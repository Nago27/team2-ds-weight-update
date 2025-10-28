#pragma once
#include <random>
#include "Matrix.hpp"

namespace vsnn {
    class Initializer {
    public:
        static void Uniform(Matrix& W, float scale = 0.01f, uint64_t seed = 42) {
            std::mt19937_64 gen(seed);
            std::uniform_real_distribution<float> dist(-scale, scale);
            for (i32 i = 0; i < W.size(); ++i) {
                *(W.data() + i) = dist(gen);
            }
        }

        static void Normal(Matrix& W, float stddev = 0.01f, uint64_t seed = 42) {
            std::mt19937_64 gen(seed);
            std::normal_distribution<float> dist(0.0f, stddev);
            for (i32 i = 0; i < W.size(); ++i) {
                *(W.data() + i) = dist(gen);
            }
        }
    };
}
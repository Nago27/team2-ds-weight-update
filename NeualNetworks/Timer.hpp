// =============================
// include/vsnn/Timer.hpp (올바른 버전)
// =============================
#pragma once

// C++에서 고정밀 시간을 측정하기 위한 표준 라이브러리입니다.
#include <chrono>

namespace vsnn {

    // 코드 실행 시간을 측정하기 위한 간단한 타이머 클래스입니다.
    class Timer {
    private:
        // 시간 측정의 시작점을 저장할 변수입니다.
        std::chrono::high_resolution_clock::time_point start_time_;

    public:
        // Tic() 메소드: 타이머를 시작시키는 역할을 합니다.
        void Tic() {
            start_time_ = std::chrono::high_resolution_clock::now();
        }

        // TocMs() 메소드: Tic() 이후부터 현재까지 몇 밀리초(ms)가 지났는지 계산하여 반환합니다.
        double TocMs() const {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = end_time - start_time_;
            // 시간 차이를 double 타입의 밀리초(milliseconds) 단위로 변환하여 반환합니다.
            return std::chrono::duration<double, std::milli>(duration).count();
        }
    }; // <-- 클래스 정의의 끝을 알리는 이 세미콜론(;)이 매우 중요합니다!

} // namespace vsnn
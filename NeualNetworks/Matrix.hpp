#pragma once
#include <Eigen/Dense>
#include <cstdint>

namespace vsnn {
    using f32 = float;
    using i32 = int32_t;

    // Matrix 타입을 Eigen의 동적 행렬에 대한 별칭(alias)으로 지정합니다.
    // RowMajor는 C-스타일 배열처럼 행을 기준으로 데이터를 저장하여 다른 라이브러리와의 호환성을 높입니다.
    using Matrix = Eigen::Matrix< //별칭 주의
        f32,
        Eigen::Dynamic,
        Eigen::Dynamic,
        Eigen::RowMajor
    >;
} //1차 커밋
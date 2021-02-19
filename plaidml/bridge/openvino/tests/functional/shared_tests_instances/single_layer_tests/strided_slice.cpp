// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/strided_slice.hpp"

using namespace LayerTestsDefinitions;

namespace {

std::vector<StridedSliceSpecificParams> ss_only_test_cases = {
    StridedSliceSpecificParams{
        {128, 1},
        {0, 0, 0},
        {0, 0, 0},
        {1, 1, 1},
        {0, 1, 1},
        {0, 1, 1},
        {1, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
    },
    StridedSliceSpecificParams{
        {128, 1},
        {0, 0, 0},
        {0, 0, 0},
        {1, 1, 1},
        {1, 0, 1},
        {1, 0, 1},
        {0, 1, 0},
        {0, 0, 0},
        {0, 0, 0},
    },
    StridedSliceSpecificParams{
        {1, 12, 100},
        {0, -1, 0},
        {0, 0, 0},
        {1, 1, 1},
        {1, 0, 1},
        {1, 0, 1},
        {0, 0, 0},
        {0, 1, 0},
        {0, 0, 0},
    },
    StridedSliceSpecificParams{
        {1, 12, 100},
        {0, 9, 0},
        {0, 11, 0},
        {1, 1, 1},
        {1, 0, 1},
        {1, 0, 1},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
    },
    StridedSliceSpecificParams{
        {1, 12, 100},
        {0, 1, 0},
        {0, -1, 0},
        {1, 1, 1},
        {1, 0, 1},
        {1, 0, 1},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
    },
    StridedSliceSpecificParams{
        {1, 12, 100},
        {0, 9, 0},
        {0, 7, 0},
        {-1, -1, -1},
        {1, 0, 1},
        {1, 0, 1},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
    },
    StridedSliceSpecificParams{
        {1, 12, 100},
        {0, 7, 0},
        {0, 9, 0},
        {-1, 1, -1},
        {1, 0, 1},
        {1, 0, 1},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
    },
    StridedSliceSpecificParams{
        {1, 12, 100},
        {0, 4, 0},
        {0, 9, 0},
        {-1, 2, -1},
        {1, 0, 1},
        {1, 0, 1},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
    },
    StridedSliceSpecificParams{
        {1, 12, 100},
        {0, 4, 0},
        {0, 10, 0},
        {-1, 2, -1},
        {1, 0, 1},
        {1, 0, 1},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
    },
    StridedSliceSpecificParams{
        {1, 12, 100},
        {0, 9, 0},
        {0, 4, 0},
        {-1, -2, -1},
        {1, 0, 1},
        {1, 0, 1},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
    },
    StridedSliceSpecificParams{
        {1, 12, 100},
        {0, 10, 0},
        {0, 4, 0},
        {-1, -2, -1},
        {1, 0, 1},
        {1, 0, 1},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
    },
    StridedSliceSpecificParams{
        {1, 12, 100},
        {0, 11, 0},
        {0, 0, 0},
        {-1, -2, -1},
        {1, 0, 1},
        {1, 0, 1},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
    },
    StridedSliceSpecificParams{
        {1, 12, 100},
        {0, -6, 0},
        {0, -8, 0},
        {-1, -2, -1},
        {1, 0, 1},
        {1, 0, 1},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
    },
    StridedSliceSpecificParams{
        {1, 12, 100, 1, 1},
        {0, -1, 0, 0},
        {0, 0, 0, 0},
        {1, 1, 1, 1},
        {1, 0, 1, 0},
        {1, 0, 1, 0},
        {},
        {0, 1, 0, 1},
        {},
    },
    StridedSliceSpecificParams{
        {2, 2, 2, 2},
        {0, 0, 0, 0},
        {2, 2, 2, 2},
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        {},
        {},
        {},
    },
    StridedSliceSpecificParams{
        {2, 2, 2, 2},
        {1, 1, 1, 1},
        {2, 2, 2, 2},
        {1, 1, 1, 1},
        {0, 0, 0, 0},
        {1, 1, 1, 1},
        {},
        {},
        {},
    },
    StridedSliceSpecificParams{
        {2, 2, 2, 2},
        {1, 1, 1, 1},
        {2, 2, 2, 2},
        {1, 1, 1, 1},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {},
        {},
        {},
    },
    StridedSliceSpecificParams{
        {2, 2, 4, 3},
        {0, 0, 0, 0},
        {2, 2, 4, 3},
        {1, 1, 2, 1},
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        {},
        {},
        {},
    },
    StridedSliceSpecificParams{
        {2, 2, 4, 2},
        {1, 0, 0, 1},
        {2, 2, 4, 2},
        {1, 1, 2, 1},
        {0, 1, 1, 0},
        {1, 1, 0, 0},
        {},
        {},
        {},
    },
    StridedSliceSpecificParams{
        {1, 2, 4, 2},
        {1, 0, 0, 0},
        {1, 2, 4, 2},
        {1, 1, -2, -1},
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        {},
        {},
        {},
    },
    StridedSliceSpecificParams{
        {2, 2, 4, 2},
        {1, 0, 0, 0},
        {1, 2, 4, 2},
        {1, 1, -2, -1},
        {0, 1, 1, 1},
        {1, 1, 1, 1},
        {},
        {},
        {},
    },
    StridedSliceSpecificParams{
        {2, 3, 4, 5, 6},
        {0, 1, 0, 0, 0},
        {2, 3, 4, 5, 6},
        {1, 1, 1, 1, 1},
        {1, 0, 1, 1, 1},
        {1, 0, 1, 1, 1},
        {},
        {0, 1, 0, 0, 0},
        {},
    },

    /// those two case will run into segment fault, it is caused by openvino inference,
    /// which is locate in 'MakeNextStageTask' of 'ie_infer_async_request_thread_safe_default.hpp'
    //    StridedSliceSpecificParams{
    //        {10, 12},
    //        {-1, 1},
    //        {-9999, 0},
    //        {-1, 1},
    //        {0, 1},
    //        {0, 1},
    //        {0, 0},
    //        {0, 0},
    //        {0, 0},
    //    },
    //    StridedSliceSpecificParams{
    //        {5, 5, 5, 5},
    //        {-1, 0, -1, 0},
    //        {-50, 0, -60, 0},
    //        {-1, 1, -1, 1},
    //        {0, 0, 0, 0},
    //        {0, 1, 0, 1},
    //        {0, 0, 0, 0},
    //        {0, 0, 0, 0},
    //        {0, 0, 0, 0},
    //    },
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_CASE_P(smoke, StridedSliceLayerTest,
                        ::testing::Combine(                                              //
                            ::testing::ValuesIn(ss_only_test_cases),                     //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
                            ::testing::Values(std::map<std::string, std::string>())),    //
                        StridedSliceLayerTest::getTestCaseName);

}  // namespace

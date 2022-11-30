// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/detection_output.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct DetectionOutputParams {
    template <class IT>
    DetectionOutputParams(const int num_classes,
                          const int background_label_id,
                          const int top_k,
                          const bool variance_encoded_in_target,
                          const std::vector<int> keep_top_k,
                          const std::string code_type,
                          const bool share_location,
                          const float nms_threshold,
                          const float confidence_threshold,
                          const bool clip_after_nms,
                          const bool clip_before_nms,
                          const bool decrease_label_id,
                          const bool normalized,
                          const size_t input_height,
                          const size_t input_width,
                          const float objectness_score,
                          const size_t num_prior_boxes,
                          const size_t num_images,
                          const bool is_priors_patch_size_1,
                          const ov::element::Type& iType,
                          const std::vector<IT>& locValues,
                          const std::vector<IT>& confValues,
                          const std::vector<IT>& priorBoxesValues,
                          const std::vector<IT>& oValues,
                          const std::string& test_name = "")
        : inType(iType),
          locData(CreateTensor(iType, locValues)),
          confData(CreateTensor(iType, confValues)),
          priorBoxesData(CreateTensor(iType, priorBoxesValues)),
          refData(CreateTensor(iType, oValues)),
          testcaseName(test_name) {
              attrs.num_classes = num_classes;
              attrs_v8.background_label_id = attrs.background_label_id = background_label_id;
              attrs_v8.top_k = attrs.top_k = top_k;
              attrs_v8.variance_encoded_in_target = attrs.variance_encoded_in_target = variance_encoded_in_target;
              attrs_v8.keep_top_k = attrs.keep_top_k = keep_top_k;
              attrs_v8.code_type = attrs.code_type = code_type;
              attrs_v8.share_location = attrs.share_location = share_location;
              attrs_v8.nms_threshold = attrs.nms_threshold = nms_threshold;
              attrs_v8.confidence_threshold = attrs.confidence_threshold = confidence_threshold;
              attrs_v8.clip_after_nms = attrs.clip_after_nms = clip_after_nms;
              attrs_v8.clip_before_nms = attrs.clip_before_nms = clip_before_nms;
              attrs_v8.decrease_label_id = attrs.decrease_label_id = decrease_label_id;
              attrs_v8.normalized = attrs.normalized = normalized;
              attrs_v8.input_height = attrs.input_height = input_height;
              attrs_v8.input_width = attrs.input_width = input_width;
              attrs_v8.objectness_score = attrs.objectness_score = objectness_score;

              size_t num_loc_classes = attrs.share_location ? 1 : attrs.num_classes;
              size_t prior_box_size = attrs.normalized ? 4 : 5;

              locShape = ov::Shape{num_images, num_prior_boxes * num_loc_classes * prior_box_size};
              confShape = ov::Shape{num_images, num_prior_boxes * attrs.num_classes};
              priorBoxesShape =
              ov::Shape{is_priors_patch_size_1 ? 1UL : num_images, attrs.variance_encoded_in_target ? 1UL : 2UL, num_prior_boxes * prior_box_size};
          }

template <class IT>
    DetectionOutputParams(const int num_classes,
                          const int background_label_id,
                          const int top_k,
                          const bool variance_encoded_in_target,
                          const std::vector<int> keep_top_k,
                          const std::string code_type,
                          const bool share_location,
                          const float nms_threshold,
                          const float confidence_threshold,
                          const bool clip_after_nms,
                          const bool clip_before_nms,
                          const bool decrease_label_id,
                          const bool normalized,
                          const size_t input_height,
                          const size_t input_width,
                          const float objectness_score,
                          const size_t num_prior_boxes,
                          const size_t num_images,
                          const bool is_priors_patch_size_1,
                          const ov::element::Type& iType,
                          const std::vector<IT>& locValues,
                          const std::vector<IT>& confValues,
                          const std::vector<IT>& priorBoxesValues,
                          const std::vector<IT>& oValues,
                          const std::vector<IT>& auxLocValues,
                          const std::vector<IT>& auxConfValues,
                          const std::string& test_name = "")
        : inType(iType),
          locData(CreateTensor(iType, locValues)),
          confData(CreateTensor(iType, confValues)),
          priorBoxesData(CreateTensor(iType, priorBoxesValues)),
          refData(CreateTensor(iType, oValues)),
          auxLocData(CreateTensor(iType, auxLocValues)),
          auxConfData(CreateTensor(iType, auxConfValues)),
          testcaseName(test_name) {
              attrs.num_classes = num_classes;
              attrs_v8.background_label_id = attrs.background_label_id = background_label_id;
              attrs_v8.top_k = attrs.top_k = top_k;
              attrs_v8.variance_encoded_in_target = attrs.variance_encoded_in_target = variance_encoded_in_target;
              attrs_v8.keep_top_k = attrs.keep_top_k = keep_top_k;
              attrs_v8.code_type = attrs.code_type = code_type;
              attrs_v8.share_location = attrs.share_location = share_location;
              attrs_v8.nms_threshold = attrs.nms_threshold = nms_threshold;
              attrs_v8.confidence_threshold = attrs.confidence_threshold = confidence_threshold;
              attrs_v8.clip_after_nms = attrs.clip_after_nms = clip_after_nms;
              attrs_v8.clip_before_nms = attrs.clip_before_nms = clip_before_nms;
              attrs_v8.decrease_label_id = attrs.decrease_label_id = decrease_label_id;
              attrs_v8.normalized = attrs.normalized = normalized;
              attrs_v8.input_height = attrs.input_height = input_height;
              attrs_v8.input_width = attrs.input_width = input_width;
              attrs_v8.objectness_score = attrs.objectness_score = objectness_score;

              size_t num_loc_classes = attrs.share_location ? 1 : attrs.num_classes;
              size_t prior_box_size = attrs.normalized ? 4 : 5;

              locShape = ov::Shape{num_images, num_prior_boxes * num_loc_classes * prior_box_size};
              confShape = ov::Shape{num_images, num_prior_boxes * attrs.num_classes};
              priorBoxesShape =
              ov::Shape{is_priors_patch_size_1 ? 1UL : num_images, attrs.variance_encoded_in_target ? 1UL : 2UL, num_prior_boxes * prior_box_size};
              auxLocShape = locShape;
              auxConfShape = confShape;
          }

    ov::op::v0::DetectionOutput::Attributes attrs;
    ov::op::v8::DetectionOutput::Attributes attrs_v8;
    ov::PartialShape locShape;
    ov::PartialShape confShape;
    ov::PartialShape priorBoxesShape;
    ov::PartialShape auxLocShape;
    ov::PartialShape auxConfShape;
    ov::element::Type inType;
    ov::Tensor locData;
    ov::Tensor confData;
    ov::Tensor priorBoxesData;
    ov::Tensor refData;
    ov::Tensor auxLocData;
    ov::Tensor auxConfData;
    std::string testcaseName;
};

class ReferenceDetectionOutputLayerTest : public testing::TestWithParam<DetectionOutputParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        if ((params.auxLocShape.size() != 0) && (params.auxConfShape.size() != 0))
            inputData = {params.locData, params.confData, params.priorBoxesData, params.auxConfData, params.auxLocData};
        else
            inputData = {params.locData, params.confData, params.priorBoxesData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<DetectionOutputParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "locShape=" << param.locShape << "_";
        result << "confShape=" << param.confShape << "_";
        result << "priorBoxesShape=" << param.priorBoxesShape << "_";
        if ((param.auxLocShape.size() != 0) && (param.auxConfShape.size() != 0)) {
            result << "auxLocShape=" << param.locShape << "_";
            result << "auxConfShape=" << param.confShape << "_";
        }
        result << "iType=" << param.inType;
        if (param.testcaseName != "")
            result << "_" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const DetectionOutputParams& params) {
        const auto loc = std::make_shared<op::v0::Parameter>(params.inType, params.locShape);
        const auto conf = std::make_shared<op::v0::Parameter>(params.inType, params.confShape);
        const auto priorBoxes = std::make_shared<op::v0::Parameter>(params.inType, params.priorBoxesShape);
        if ((params.auxLocShape.size() != 0) && (params.auxConfShape.size() != 0)) {
            const auto auxConf = std::make_shared<op::v0::Parameter>(params.inType, params.auxConfShape);
            const auto auxLoc = std::make_shared<op::v0::Parameter>(params.inType, params.auxLocShape);
            const auto DetectionOutput = std::make_shared<op::v0::DetectionOutput>(loc, conf, priorBoxes, auxConf, auxLoc, params.attrs);
            return std::make_shared<ov::Model>(NodeVector {DetectionOutput}, ParameterVector {loc, conf, priorBoxes, auxConf, auxLoc});
        } else {
            const auto DetectionOutput = std::make_shared<op::v0::DetectionOutput>(loc, conf, priorBoxes, params.attrs);
            return std::make_shared<ov::Model>(NodeVector {DetectionOutput}, ParameterVector {loc, conf, priorBoxes});
        }
    }
};

class ReferenceDetectionOutputV8LayerTest : public testing::TestWithParam<DetectionOutputParams>,
                                          public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        if ((params.auxLocShape.size() != 0) && (params.auxConfShape.size() != 0))
            inputData = {params.locData, params.confData, params.priorBoxesData, params.auxConfData, params.auxLocData};
        else
            inputData = {params.locData, params.confData, params.priorBoxesData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<DetectionOutputParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "locShape=" << param.locShape << "_";
        result << "confShape=" << param.confShape << "_";
        result << "priorBoxesShape=" << param.priorBoxesShape << "_";
        if ((param.auxLocShape.size() != 0) && (param.auxConfShape.size() != 0)) {
            result << "auxLocShape=" << param.locShape << "_";
            result << "auxConfShape=" << param.confShape << "_";
        }
        result << "iType=" << param.inType;
        if (param.testcaseName != "")
            result << "_" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const DetectionOutputParams& params) {
        const auto loc = std::make_shared<op::v0::Parameter>(params.inType, params.locShape);
        const auto conf = std::make_shared<op::v0::Parameter>(params.inType, params.confShape);
        const auto priorBoxes = std::make_shared<op::v0::Parameter>(params.inType, params.priorBoxesShape);
        if ((params.auxLocShape.size() != 0) && (params.auxConfShape.size() != 0)) {
            const auto auxConf = std::make_shared<op::v0::Parameter>(params.inType, params.auxConfShape);
            const auto auxLoc = std::make_shared<op::v0::Parameter>(params.inType, params.auxLocShape);
            const auto DetectionOutput =
                std::make_shared<op::v8::DetectionOutput>(loc, conf, priorBoxes, auxConf, auxLoc, params.attrs_v8);
            return std::make_shared<ov::Model>(NodeVector{DetectionOutput},
                                                  ParameterVector{loc, conf, priorBoxes, auxConf, auxLoc});
        } else {
            const auto DetectionOutput = std::make_shared<op::v8::DetectionOutput>(loc, conf, priorBoxes, params.attrs_v8);
            return std::make_shared<ov::Model>(NodeVector{DetectionOutput}, ParameterVector{loc, conf, priorBoxes});
        }
    }
};

TEST_P(ReferenceDetectionOutputLayerTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceDetectionOutputV8LayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<DetectionOutputParams> generateDetectionOutputFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<DetectionOutputParams> detectionOutputParams {
        DetectionOutputParams(3,
                              -1,
                              -1,
                              true,
                              {2},
                              "caffe.PriorBoxParameter.CORNER",
                              false,
                              0.5,
                              0.3,
                              false,
                              true,
                              false,
                              true,
                              0,
                              0,
                              0,
                              2,
                              2,
                              true,
                              IN_ET,
                              std::vector<T>{
                                    // batch 0, class 0
                                    0.1f, 0.1f, 0.2f, 0.2f, 0.0f, 0.1f, 0.2f, 0.15f,
                                    // batch 0, class 1
                                    0.3f, 0.2f, 0.5f, 0.3f, 0.2f, 0.1f, 0.42f, 0.66f,
                                    // batch 0, class 2
                                    0.05f, 0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.33f, 0.44f,
                                    // batch 1, class 0
                                    0.2f, 0.1f, 0.4f, 0.2f, 0.1f, 0.05f, 0.2f, 0.25f,
                                    // batch 1, class 1
                                    0.1f, 0.2f, 0.5f, 0.3f, 0.1f, 0.1f, 0.12f, 0.34f,
                                    // batch 1, class 2
                                    0.25f, 0.11f, 0.4f, 0.32f, 0.2f, 0.12f, 0.38f, 0.24f},
                              std::vector<T>{
                                    // batch 0
                                    0.1f, 0.9f, 0.4f, 0.7f, 0, 0.2f,
                                    // batch 1
                                    0.7f, 0.8f, 0.42f, 0.33f, 0.81f, 0.2f},
                              std::vector<T>{
                                    // prior box 0
                                    0.0f, 0.5f, 0.1f, 0.2f,
                                    // prior box 1
                                    0.0f, 0.3f, 0.1f, 0.35f},
                              std::vector<T>{
                                    0, 0, 0.7f,  0.2f,  0.4f,  0.52f, 1,    0, 1, 0.9f, 0,   0.6f,  0.3f, 0.35f,
                                    1, 1, 0.81f, 0.25f, 0.41f, 0.5f,  0.67f, 1, 1, 0.8f, 0.1f, 0.55f, 0.3f, 0.45f},
                                    "3_inputs"),
        DetectionOutputParams(3,
                              -1,
                              -1,
                              true,
                              {2},
                              "caffe.PriorBoxParameter.CORNER",
                              true,
                              0.5f,
                              0.3f,
                              false,
                              true,
                              false,
                              true,
                              0,
                              0,
                              0,
                              2,
                              2,
                              false,
                              IN_ET,
                              std::vector<T>{
                                    // batch 0
                                    0.1f, 0.1f, 0.2f, 0.2f, 0.0f, 0.1f, 0.2f, 0.15f,
                                    // batch 1
                                    0.2f, 0.1f, 0.4f, 0.2f, 0.1f, 0.05f, 0.2f, 0.25f},
                              std::vector<T>{
                                    // batch 0
                                    0.1f, 0.9f, 0.4f, 0.7f, 0, 0.2f,
                                    // batch 1
                                    0.7f, 0.8f, 0.42f, 0.33f, 0.81f, 0.2f},
                              std::vector<T>{
                                    // batch 0
                                    0.0f, 0.5f, 0.1f, 0.2f, 0.0f, 0.3f, 0.1f, 0.35f,
                                    // batch 1
                                    0.33f, 0.2f, 0.52f, 0.37f, 0.22f, 0.1f, 0.32f, 0.36f},
                              std::vector<T>{
                                    0, 0, 0.7f,  0,    0.4f,  0.3f,  0.5f,  0, 1, 0.9f, 0.1f,  0.6f, 0.3f,  0.4f,
                                    1, 1, 0.81f, 0.32f, 0.15f, 0.52f, 0.61f, 1, 1, 0.8f, 0.53f, 0.3f, 0.92f, 0.57f},
                                    "3_inputs_share_location"),
        DetectionOutputParams(3,
                              -1,
                              -1,
                              true,
                              {2},
                              "caffe.PriorBoxParameter.CORNER",
                              true,
                              0.5f,
                              0.3f,
                              false,
                              true,
                              false,
                              true,
                              0,
                              0,
                              0,
                              2,
                              2,
                              false,
                              IN_ET,
                              std::vector<T>{
                                    // batch 0
                                    0.1f, 0.1f, 0.2f, 0.2f, 0.0f, 0.1f, 0.2f, 0.15f,
                                    // batch 1
                                    0.2f, 0.1f, 0.4f, 0.2f, 0.1f, 0.05f, 0.2f, 0.25f},
                              std::vector<T>{
                                    // batch 0
                                    0.1f, 0.9f, 0.4f, 0.7f, 0, 0.2f,
                                    // batch 1
                                    0.7f, 0.8f, 0.42f, 0.33f, 0.81f, 0.2f},
                              std::vector<T>{
                                    // batch 0
                                    0.0f, 0.5f, 0.1f, 0.2f, 0.0f, 0.3f, 0.1f, 0.35f,
                                    // batch 1
                                    0.33f, 0.2f, 0.52f, 0.37f, 0.22f, 0.1f, 0.32f, 0.36f},
                              std::vector<T>{
                                    0, 0, 0.7f,  0,    0.4f,  0.3f,  0.5f,  0, 1, 0.9f, 0.1f,  0.6f, 0.3f,  0.4f,
                                    1, 1, 0.81f, 0.32f, 0.15f, 0.52f, 0.61f, 1, 1, 0.8f, 0.53f, 0.3f, 0.92f, 0.57f},
                                    "3_inputs_normalized"),
        DetectionOutputParams(2,
                              -1,
                              -1,
                              false,
                              {-1},
                              "caffe.PriorBoxParameter.CORNER",
                              false,
                              0.5f,
                              0.3f,
                              false,
                              true,
                              false,
                              true,
                              0,
                              0,
                              0,
                              2,
                              3,
                              false,
                              IN_ET,
                              std::vector<T>{
                                    // batch 0, class 0
                                    0.1f, 0.1f, 0.2f, 0.2f, 0.0f, 0.1f, 0.2f, 0.15f,
                                    // batch 0, class 1
                                    0.3f, 0.2f, 0.5f, 0.3f, 0.2f, 0.1f, 0.42f, 0.66f,
                                    // batch 1, class 0
                                    0.05f, 0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.33f, 0.44f,
                                    // batch 1, class 1
                                    0.2f, 0.1f, 0.4f, 0.2f, 0.1f, 0.05f, 0.2f, 0.25f,
                                    // batch 2, class 0
                                    0.1f, 0.2f, 0.5f, 0.3f, 0.1f, 0.1f, 0.12f, 0.34f,
                                    // batch 2, class 1
                                    0.25f, 0.11f, 0.4f, 0.32f, 0.2f, 0.12f, 0.38f, 0.24f},
                              std::vector<T>{
                                    // batch 0
                                    0.1f, 0.9f, 0.4f, 0.7f,
                                    // batch 1
                                    0.7f, 0.8f, 0.42f, 0.33f,
                                    // batch 1
                                    0.1f, 0.2f, 0.32f, 0.43f},
                              std::vector<T>{
                                    // batch 0 priors
                                    0.0f, 0.5f, 0.1f, 0.2f, 0.0f, 0.3f, 0.1f, 0.35f,
                                    // batch 0 variances
                                    0.12f, 0.11f, 0.32f, 0.02f, 0.02f, 0.20f, 0.09f, 0.71f,
                                    // batch 1 priors
                                    0.33f, 0.2f, 0.52f, 0.37f, 0.22f, 0.1f, 0.32f, 0.36f,
                                    // batch 1 variances
                                    0.01f, 0.07f, 0.12f, 0.13f, 0.41f, 0.33f, 0.2f, 0.1f,
                                    // batch 2 priors
                                    0.0f, 0.3f, 0.1f, 0.35f, 0.22f, 0.1f, 0.32f, 0.36f,
                                    // batch 2 variances
                                    0.32f, 0.02f, 0.13f, 0.41f, 0.33f, 0.2f, 0.02f, 0.20f},
                              std::vector<T>{
                                    0, 0, 0.4f,  0.006f, 0.34f,   0.145f,  0.563f,  0,  1, 0.9f,  0,      0.511f, 0.164f,  0.203f,
                                    0, 1, 0.7f,  0.004f, 0.32f,   0.1378f, 0.8186f, 1,  0, 0.7f,  0.3305f, 0.207f, 0.544f,  0.409f,
                                    1, 0, 0.42f, 0.302f, 0.133f,  0.4f,    0.38f,   1,  1, 0.8f,  0.332f,  0.207f, 0.5596f, 0.4272f,
                                    1, 1, 0.33f, 0.261f, 0.1165f, 0.36f,   0.385f,  2,  0, 0.32f, 0.3025f, 0.122f, 0.328f,  0.424f,
                                    2, 1, 0.43f, 0.286f, 0.124f,  0.3276f, 0.408f,  -1, 0, 0,    0,      0,     0,      0,
                                    0, 0, 0,    0,     0,      0,      0,      0,  0, 0,    0,      0,     0,      0},
                                    "3_inputs_keep_all_bboxes"),
        DetectionOutputParams(3,
                              -1,
                              -1,
                              true,
                              {2},
                              "caffe.PriorBoxParameter.CENTER_SIZE",
                              false,
                              0.5f,
                              0.3f,
                              false,
                              true,
                              false,
                              true,
                              0,
                              0,
                              0,
                              2,
                              2,
                              false,
                              IN_ET,
                              std::vector<T>{
                                    // batch 0, class 0
                                    0.1f, 0.1f, 0.2f, 0.2f, 0.0f, 0.1f, 0.2f, 0.15f,
                                    // batch 0, class 1
                                    0.3f, 0.2f, 0.5f, 0.3f, 0.2f, 0.1f, 0.42f, 0.66f,
                                    // batch 0, class 2
                                    0.05f, 0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.33f, 0.44f,
                                    // batch 1, class 0
                                    0.2f, 0.1f, 0.4f, 0.2f, 0.1f, 0.05f, 0.2f, 0.25f,
                                    // batch 1, class 1
                                    0.1f, 0.2f, 0.5f, 0.3f, 0.1f, 0.1f, 0.12f, 0.34f,
                                    // batch 1, class 2
                                    0.25f, 0.11f, 0.4f, 0.32f, 0.2f, 0.12f, 0.38f, 0.24f},
                              std::vector<T>{
                                    // batch 0
                                    0.1f, 0.9f, 0.4f, 0.7f, 0, 0.2f,
                                    // batch 1
                                    0.7f, 0.8f, 0.42f, 0.33f, 0.81f, 0.2f},
                              std::vector<T>{
                                    // batch 0
                                    0.0f, 0.5f, 0.1f, 0.2f, 0.0f, 0.3f, 0.1f, 0.35f,
                                    // batch 1
                                    0.33f, 0.2f, 0.52f, 0.37f, 0.22f, 0.1f, 0.32f, 0.36f},
                              std::vector<T>{
                                    0, 0, 0.7f,  0,          0.28163019f,  0.14609808f, 0.37836978f,
                                    0, 1, 0.9f,  0,          0.49427515f,  0.11107014f, 0.14572485f,
                                    1, 1, 0.81f, 0.22040875f, 0.079573378f, 0.36959124f, 0.4376266f,
                                    1, 1, 0.8f,  0.32796675f, 0.18435785f,  0.56003326f, 0.40264216f},
                                    "3_inputs_center_size"),
        DetectionOutputParams(2,
                              -1,
                              -1,
                              true,
                              {2},
                              "caffe.PriorBoxParameter.CORNER",
                              false,
                              0.5f,
                              0.3f,
                              false,
                              true,
                              false,
                              true,
                              0,
                              0,
                              0.6f,
                              2,
                              2,
                              false,
                              IN_ET,
                              std::vector<T>{
                                    // batch 0, class 0
                                    0.1f, 0.1f, 0.2f, 0.2f, 0.0f, 0.1f, 0.2f, 0.15f,
                                    // batch 0, class 1
                                    0.3f, 0.2f, 0.5f, 0.3f, 0.2f, 0.1f, 0.42f, 0.66f,
                                    // batch 1, class 0
                                    0.2f, 0.1f, 0.4f, 0.2f, 0.1f, 0.05f, 0.2f, 0.25f,
                                    // batch 1, class 1
                                    0.1f, 0.2f, 0.5f, 0.3f, 0.1f, 0.1f, 0.12f, 0.34f},
                              std::vector<T>{
                                    // batch 0
                                    0.1f, 0.9f, 0.4f, 0.7f,
                                    // batch 1
                                    0.42f, 0.33f, 0.81f, 0.2f},
                              std::vector<T>{
                                    // batch 0
                                    0.0f, 0.5f, 0.1f, 0.2f, 0.0f, 0.3f, 0.1f, 0.35f,
                                    // batch 1
                                    0.33f, 0.2f, 0.52f, 0.37f, 0.22f, 0.1f, 0.32f, 0.36f},
                              std::vector<T>{
                                    0, 0, 0.4f,  0.55f, 0.61f, 1, 0.97f, 0, 1, 0.7f,  0.4f,  0.52f, 0.9f, 1,
                                    1, 0, 0.42f, 0.83f, 0.5f,  1, 0.87f, 1, 1, 0.33f, 0.63f, 0.35f, 1,   1},
                              std::vector<T>{
                                    // batch 0, class 0
                                    0.1f, 0.2f, 0.5f, 0.3f, 0.1f, 0.1f, 0.12f, 0.34f,
                                    // batch 0, class 1
                                    0.25f, 0.11f, 0.4f, 0.32f, 0.2f, 0.12f, 0.38f, 0.24f,
                                    // batch 1, class 0
                                    0.3f, 0.2f, 0.5f, 0.3f, 0.2f, 0.1f, 0.42f, 0.66f,
                                    // batch 1, class 1
                                    0.05f, 0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.33f, 0.44f},
                              std::vector<T>{
                                    // batch 0
                                    0.1f, 0.3f, 0.5f, 0.8f,
                                    // batch 1
                                    0.5f, 0.8f, 0.01f, 0.1f},
                                    "5_inputs"),
    };
    return detectionOutputParams;
}

std::vector<DetectionOutputParams> generateDetectionOutputCombinedParams() {
    const std::vector<std::vector<DetectionOutputParams>> detectionOutputTypeParams {
        generateDetectionOutputFloatParams<element::Type_t::f64>(),
        generateDetectionOutputFloatParams<element::Type_t::f32>(),
        generateDetectionOutputFloatParams<element::Type_t::f16>(),
        generateDetectionOutputFloatParams<element::Type_t::bf16>(),
        };
    std::vector<DetectionOutputParams> combinedParams;

    for (const auto& params : detectionOutputTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_DetectionOutput_With_Hardcoded_Refs, ReferenceDetectionOutputLayerTest,
    testing::ValuesIn(generateDetectionOutputCombinedParams()), ReferenceDetectionOutputLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DetectionOutput_With_Hardcoded_Refs,
                         ReferenceDetectionOutputV8LayerTest,
                         testing::ValuesIn(generateDetectionOutputCombinedParams()),
                         ReferenceDetectionOutputV8LayerTest::getTestCaseName);

} // namespace

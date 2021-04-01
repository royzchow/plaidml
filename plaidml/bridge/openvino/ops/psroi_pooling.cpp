// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using namespace edsl;             // NOLINT[build/namespaces]

namespace {

template <typename T>
std::vector<T> cast_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto* ngraph_const = ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(operand_idx));
  if (ngraph_const) {
    return ngraph_const->cast_vector<T>();
  } else {
    THROW_IE_EXCEPTION
        << "Dynamic coordinate is not currently supported by PlaidML plugin; all coordinate must be Constants. ";
  }
}

// Get single bin output
edsl::Tensor single_bin_pooling(edsl::Tensor I, size_t ph, size_t pw, size_t group_size) {
  std::vector<edsl::TensorDim> I_dims(4), O_dims(4);
  std::vector<edsl::TensorIndex> I_idxs(4), O_idxs(4);
  I.bind_dims(I_dims);
  O_dims[0] = O_dims[2] = O_dims[3] = edsl::TensorDim(1);
  O_dims[1] = I_dims[1] / (group_size * group_size);
  I_idxs[1] = O_idxs[1] * group_size * group_size + ph * group_size + pw;
  edsl::Tensor O = (Contraction(O_dims, O_idxs).sum(I(I_idxs))) / (I_dims[2] * I_dims[3]);
  return O;
}

}  // namespace

namespace PlaidMLPlugin {

const static int BOX_ELEMENT_SIZE = 5;  // NOLINT

void registerPSROIPooling() {
  registerOp("PSROIPooling", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset5::PSROIPooling>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);  // Input

    std::vector<float> coords = cast_constant_operand<float>(1, layer);
    IE_ASSERT((coords.size() % BOX_ELEMENT_SIZE) == 0);
    // Get attributes about the operation.
    auto I_shape = layer->get_input_shape(0);
    auto output_dim = layer->get_output_dim();
    auto group_size = layer->get_group_size();
    auto spatial_scale = layer->get_spatial_scale();
    auto spatial_bins_x = layer->get_spatial_bins_x();
    auto spatial_bins_y = layer->get_spatial_bins_y();
    auto mode = layer->get_mode();
    auto channel_in = I_shape[1];
    auto height = I_shape[2];
    auto width = I_shape[3];
    auto num_rois = coords.size() / BOX_ELEMENT_SIZE;
    auto num_classes = output_dim;
    auto pooling_height = group_size;
    auto pooling_width = group_size;
    edsl::Tensor output;
    if (mode == "average") {
      if (output_dim * group_size * group_size != channel_in) {
        THROW_IE_EXCEPTION << "Incorrected channel of the input tensor.";
      }
      std::vector<edsl::Tensor> bin_output_concat_batch_size_vec;
      for (size_t roi = 0; roi < num_rois; ++roi) {
        // Get the start and end coordinate of the box.
        auto batch_id = coords[roi * BOX_ELEMENT_SIZE];
        auto start_w = coords[roi * BOX_ELEMENT_SIZE + 1];
        auto start_h = coords[roi * BOX_ELEMENT_SIZE + 2];
        auto end_w = coords[roi * BOX_ELEMENT_SIZE + 3];
        auto end_h = coords[roi * BOX_ELEMENT_SIZE + 4];
        start_w = (std::roundf(start_w)) * spatial_scale;
        start_h = (std::roundf(start_h)) * spatial_scale;
        end_w = (std::roundf(end_w) + 1.0f) * spatial_scale;
        end_h = (std::roundf(end_h) + 1.0f) * spatial_scale;
        auto box_width = end_w - start_w;
        auto box_height = end_h - start_h;
        auto bin_width = box_width / pooling_width;
        auto bin_height = box_height / pooling_height;

        std::vector<edsl::Tensor> bin_output_concat_width_vec;
        for (size_t ph = 0; ph < pooling_height; ++ph) {
          std::vector<edsl::Tensor> single_bin_output_vec;
          for (size_t pw = 0; pw < pooling_width; ++pw) {
            size_t bin_start_w = std::min(static_cast<size_t>(floorf(start_w + pw * bin_width)), width - 1);
            size_t bin_start_h = std::min(static_cast<size_t>(floorf(start_h + ph * bin_height)), height - 1);
            size_t bin_end_w = std::min(static_cast<size_t>(ceilf(start_w + (pw + 1) * bin_width)), width);
            size_t bin_end_h = std::min(static_cast<size_t>(ceilf(start_h + (ph + 1) * bin_height)), height);
            auto bin_input = op::slice(I)
                                 .add_dim(static_cast<size_t>(batch_id), static_cast<size_t>(batch_id + 1))
                                 .add_dim(0, channel_in)
                                 .add_dim(bin_start_h, bin_end_h)
                                 .add_dim(bin_start_w, bin_end_w);
            auto single_bin_output = single_bin_pooling(bin_input, ph, pw, group_size);
            single_bin_output_vec.push_back(single_bin_output);
          }
          auto bin_output_concat_width = op::concatenate(single_bin_output_vec, 3);
          bin_output_concat_width_vec.push_back(bin_output_concat_width);
        }
        auto bin_output_concat_height = op::concatenate(bin_output_concat_width_vec, 2);
        bin_output_concat_batch_size_vec.push_back(bin_output_concat_height);
      }
      output = op::concatenate(bin_output_concat_batch_size_vec, 0);
    } else if (mode == "bilinear") {
      if (output_dim * spatial_bins_x * spatial_bins_y != channel_in) {
        THROW_IE_EXCEPTION << "Incorrected channel of the input tensor.";
      }
      std::vector<edsl::Tensor> bin_concat_channel_vec;
      for (size_t roi = 0; roi < num_rois; ++roi) {
        // Get the start and end coordinate of the box.
        auto batch_id = coords[roi * BOX_ELEMENT_SIZE];
        auto start_w = coords[roi * BOX_ELEMENT_SIZE + 1];
        auto start_h = coords[roi * BOX_ELEMENT_SIZE + 2];
        auto end_w = coords[roi * BOX_ELEMENT_SIZE + 3];
        auto end_h = coords[roi * BOX_ELEMENT_SIZE + 4];
        start_w = start_w * spatial_scale;
        start_h = start_h * spatial_scale;
        end_w = end_w * spatial_scale;
        end_h = end_h * spatial_scale;
        auto box_width = end_w - start_w;
        auto box_height = end_h - start_h;
        auto bin_width = box_width / spatial_bins_x;
        auto bin_height = box_height / spatial_bins_y;
        float width_scale = 0;
        float height_scale = 0;
        if (pooling_width > 1) width_scale = bin_width * (width - 1) / (pooling_width - 1);
        if (pooling_height > 1) height_scale = bin_height * (height - 1) / (pooling_height - 1);
        std::vector<edsl::Tensor> bin_concat_height_vec;
        for (size_t c_out = 0; c_out < output_dim; c_out++) {
          std::vector<edsl::Tensor> bin_concat_width_vec;
          for (size_t ph = 0; ph < pooling_height; ph++) {
            std::vector<edsl::Tensor> single_bin_vec;
            for (size_t pw = 0; pw < pooling_width; pw++) {
              auto single_bin = edsl::cast(edsl::index({edsl::TensorDim(1)}, 0), DType::FLOAT32);
              for (size_t sby = 0; sby < spatial_bins_y; sby++) {
                for (size_t sbx = 0; sbx < spatial_bins_x; sbx++) {
                  auto bin_start_w = start_w + sbx * bin_width;
                  auto bin_start_h = start_h + sby * bin_height;

                  auto point_x = pooling_width > 1 ? (pw * width_scale + bin_start_w * (width - 1))
                                                   : (bin_start_w + bin_start_w + bin_width) * (width - 1) / 2;
                  auto point_y = pooling_height > 1 ? (ph * height_scale + bin_start_h * (height - 1))
                                                    : (bin_start_h + bin_start_h + bin_height) * (height - 1) / 2;
                  if (point_x < width && point_y < height) {
                    auto c_in = edsl::cast(edsl::index({edsl::TensorDim(1)}, 0), DType::FLOAT32) +
                                (sby * spatial_bins_x + sbx) * num_classes + c_out;
                    auto batch_id_tensor = edsl::cast(edsl::index({edsl::TensorDim(1)}, 0), DType::FLOAT32) + batch_id;
                    auto point_x_tensor = edsl::cast(edsl::index({edsl::TensorDim(1)}, 0), DType::FLOAT32) + point_x;
                    auto point_y_tensor = edsl::cast(edsl::index({edsl::TensorDim(1)}, 0), DType::FLOAT32) + point_y;
                    // Get the single sub bin with bilinear interpolate.
                    edsl::Tensor single_sub_bin = edsl::gather(I, batch_id_tensor).axis(0);
                    single_sub_bin = edsl::gather(single_sub_bin, c_in).axis(1);
                    single_sub_bin = edsl::gather(single_sub_bin, point_y_tensor).axis(2);
                    single_sub_bin = edsl::gather(single_sub_bin, point_x_tensor).axis(3);
                    single_bin = single_bin + single_sub_bin;
                  }
                }
              }
              single_bin = single_bin / (spatial_bins_x * spatial_bins_y);
              single_bin = op::reshape(single_bin, make_tuple<int64_t>({1, 1, 1, 1}));
              single_bin_vec.push_back(single_bin);
            }
            auto bin_concat_width = op::concatenate(single_bin_vec, 3);
            bin_concat_width_vec.push_back(bin_concat_width);
          }
          auto bin_concat_height = op::concatenate(bin_concat_width_vec, 2);
          bin_concat_height_vec.push_back(bin_concat_height);
        }
        auto bin_concat_channel = op::concatenate(bin_concat_height_vec, 1);
        bin_concat_channel_vec.push_back(bin_concat_channel);
      }
      output = op::concatenate(bin_concat_channel_vec, 0);
    } else {
      THROW_IE_EXCEPTION << "Invalid PS ROI pooling mode.";
    }
    return edsl::make_tuple(output);
  });
}

}  // namespace PlaidMLPlugin

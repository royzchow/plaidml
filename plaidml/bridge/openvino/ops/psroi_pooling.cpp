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

edsl::Tensor extract_reduce_rois_dim(edsl::Tensor I) {
  std::vector<edsl::TensorDim> I_dims(10), O_dims(8);
  std::vector<edsl::TensorIndex> I_idxs(10), O_idxs(8);
  I.bind_dims(I_dims);
  for (size_t i = 0; i < 8; ++i) {
    O_dims[i] = I_dims[i + 2];
  }
  for (size_t i = 0; i < 8; ++i) {
    I_idxs[i + 2] = O_idxs[i];
  }
  I_idxs[0] = I_idxs[2];
  edsl::Tensor O = Contraction(O_dims, O_idxs).assign(I(I_idxs));
  return O;
}

edsl::Tensor extract_reduce_other_dims(edsl::Tensor I) {
  std::vector<edsl::TensorDim> I_dims(13), O_dims(7);
  std::vector<edsl::TensorIndex> I_idxs(13), O_idxs(7);
  I.bind_dims(I_dims);
  for (size_t i = 0; i < 7; ++i) {
    O_dims[i] = I_dims[i + 6];
  }
  for (size_t i = 0; i < 7; ++i) {
    I_idxs[i + 6] = O_idxs[i];
  }
  for (size_t i = 0; i < 6; ++i) {
    I_idxs[i] = I_idxs[i + 6];
  }
  edsl::Tensor O = Contraction(O_dims, O_idxs).assign(I(I_idxs));
  return O;
}

edsl::Tensor extract_tensor(edsl::Tensor I) {
  std::vector<edsl::TensorDim> I_dims(12), O_dims(6);
  std::vector<edsl::TensorIndex> I_idxs(12), O_idxs(6);
  I.bind_dims(I_dims);
  for (size_t i = 0; i < 6; ++i) {
    O_dims[i] = I_dims[i];
  }
  for (size_t i = 0; i < 6; ++i) {
    I_idxs[i] = O_idxs[i];
  }
  for (size_t i = 0; i < 6; ++i) {
    I_idxs[i + 6] = I_idxs[i];
  }
  edsl::Tensor O = Contraction(O_dims, O_idxs).assign(I(I_idxs));
  return O;
}

edsl::Tensor compute_psroi_pooling(edsl::Tensor I) {
  std::vector<edsl::TensorDim> I_dims(6), O_dims(4);
  std::vector<edsl::TensorIndex> I_idxs(6), O_idxs(4);
  I.bind_dims(I_dims);
  for (size_t i = 0; i < 4; ++i) {
    O_dims[i] = I_dims[i];
  }
  for (size_t i = 0; i < 4; ++i) {
    I_idxs[i] = O_idxs[i];
  }
  edsl::Tensor O = Contraction(O_dims, O_idxs).sum(I(I_idxs));
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
    auto C = ctx.operands.at(1);  // Coord

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
      // Get the start and end coordinate of the box.
      edsl::Tensor batch_id = op::slice(C).add_dim(0, num_rois).add_dim(0, 1);
      edsl::Tensor start_w = op::slice(C).add_dim(0, num_rois).add_dim(1, 2);
      edsl::Tensor start_h = op::slice(C).add_dim(0, num_rois).add_dim(2, 3);
      edsl::Tensor end_w = op::slice(C).add_dim(0, num_rois).add_dim(3, 4);
      edsl::Tensor end_h = op::slice(C).add_dim(0, num_rois).add_dim(4, 5);
      start_w = start_w * spatial_scale;
      start_h = start_h * spatial_scale;
      end_w = end_w * spatial_scale;
      end_h = end_h * spatial_scale;
      edsl::Tensor box_width = end_w - start_w;
      edsl::Tensor box_height = end_h - start_h;
      edsl::Tensor bin_width = box_width / spatial_bins_x;
      edsl::Tensor bin_height = box_height / spatial_bins_y;
      edsl::Tensor width_scale = edsl::cast(edsl::index({edsl::TensorDim(1)}, 0), DType::FLOAT32);
      edsl::Tensor height_scale = edsl::cast(edsl::index({edsl::TensorDim(1)}, 0), DType::FLOAT32);
      if (pooling_width > 1) width_scale = bin_width * (width - 1) / (pooling_width - 1);
      if (pooling_height > 1) height_scale = bin_height * (height - 1) / (pooling_height - 1);

      start_w = op::reshape(start_w, make_tuple<size_t>({num_rois, 1, 1, 1, 1, 1}));
      start_h = op::reshape(start_h, make_tuple<size_t>({num_rois, 1, 1, 1, 1, 1}));
      bin_width = op::reshape(bin_width, make_tuple<size_t>({num_rois, 1, 1, 1, 1, 1}));
      bin_height = op::reshape(bin_height, make_tuple<size_t>({num_rois, 1, 1, 1, 1, 1}));
      width_scale = op::reshape(width_scale, make_tuple<size_t>({num_rois, 1, 1, 1, 1, 1}));
      height_scale = op::reshape(height_scale, make_tuple<size_t>({num_rois, 1, 1, 1, 1, 1}));

      edsl::Tensor c_out = edsl::cast(edsl::index({edsl::TensorDim(num_rois), edsl::TensorDim(output_dim),
                                                   edsl::TensorDim(pooling_height), edsl::TensorDim(pooling_width),
                                                   edsl::TensorDim(spatial_bins_y), edsl::TensorDim(spatial_bins_x)},
                                                  1),
                                      DType::FLOAT32);
      edsl::Tensor ph = edsl::cast(edsl::index({edsl::TensorDim(num_rois), edsl::TensorDim(output_dim),
                                                edsl::TensorDim(pooling_height), edsl::TensorDim(pooling_width),
                                                edsl::TensorDim(spatial_bins_y), edsl::TensorDim(spatial_bins_x)},
                                               2),
                                   DType::FLOAT32);
      edsl::Tensor pw = edsl::cast(edsl::index({edsl::TensorDim(num_rois), edsl::TensorDim(output_dim),
                                                edsl::TensorDim(pooling_height), edsl::TensorDim(pooling_width),
                                                edsl::TensorDim(spatial_bins_y), edsl::TensorDim(spatial_bins_x)},
                                               3),
                                   DType::FLOAT32);
      edsl::Tensor sby = edsl::cast(edsl::index({edsl::TensorDim(num_rois), edsl::TensorDim(output_dim),
                                                 edsl::TensorDim(pooling_height), edsl::TensorDim(pooling_width),
                                                 edsl::TensorDim(spatial_bins_y), edsl::TensorDim(spatial_bins_x)},
                                                4),
                                    DType::FLOAT32);
      edsl::Tensor sbx = edsl::cast(edsl::index({edsl::TensorDim(num_rois), edsl::TensorDim(output_dim),
                                                 edsl::TensorDim(pooling_height), edsl::TensorDim(pooling_width),
                                                 edsl::TensorDim(spatial_bins_y), edsl::TensorDim(spatial_bins_x)},
                                                5),
                                    DType::FLOAT32);
      edsl::Tensor bin_start_w = start_w + sbx * bin_width;
      edsl::Tensor bin_start_h = start_h + sby * bin_height;
      edsl::Tensor point_x = pooling_width > 1 ? (pw * width_scale + bin_start_w * (width - 1))
                                               : (bin_start_w + bin_start_w + bin_width) * (width - 1) / 2;
      edsl::Tensor point_y = pooling_height > 1 ? (ph * height_scale + bin_start_h * (height - 1))
                                                : (bin_start_h + bin_start_h + bin_height) * (height - 1) / 2;
      edsl::Tensor c_in = (sby * spatial_bins_x + sbx) * num_classes + c_out;

      output = edsl::gather(I, batch_id).axis(0);
      output = edsl::gather(output, c_in).axis(-3);
      output = extract_reduce_rois_dim(output);
      output = edsl::gather(output, point_y).axis(-2);
      output = extract_reduce_other_dims(output);
      output = edsl::gather(output, point_x).axis(-1);
      output = extract_tensor(output);
      output = compute_psroi_pooling(output);
      output = output / (spatial_bins_x * spatial_bins_y);

    } else {
      THROW_IE_EXCEPTION << "Invalid PS ROI pooling mode.";
    }
    return edsl::make_tuple(output);
  });
}

}  // namespace PlaidMLPlugin

////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#define LBANN_BATCH_NORMALIZATION_LAYER_INSTANTIATE
#include "lbann/layers/regularizers/batch_normalization.hpp"
#include "lbann/execution_contexts/sgd_execution_context.hpp"

namespace lbann {

template <typename TensorDataType>
void fp_compute_impl(batch_normalization_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>& l) {
  constexpr TensorDataType zero = 0;
  constexpr TensorDataType one = 1;
  const bool is_training = l.m_model->get_execution_context().get_execution_mode() == execution_mode::training;

  // Matrices
  const auto& input = l.get_prev_activations();
  const auto& local_input = input.LockedMatrix();
  auto& local_output = l.get_local_activations();

  // Matrix parameters
  const auto& width = input.Width();
  const auto& local_width = local_input.Width();
  const auto& output_dims = l.get_output_dims();
  const auto& num_channels = output_dims[0];
  const auto& channel_size = l.get_output_size() / num_channels;

  // Compute statistics
  if (is_training) {

    // Local matrices
    auto& local_mean = l.m_mean_v->Matrix();
    auto& local_var = l.m_var_v->Matrix();
    auto& local_running_mean = l.m_weights[2]->get_values().Matrix();
    auto& local_running_var = l.m_weights[3]->get_values().Matrix();

    // Compute sums and sums of squares
    LBANN_OMP_PARALLEL_FOR
    for (El::Int channel = 0; channel < num_channels; ++channel) {
      TensorDataType sum = zero;
      TensorDataType sqsum = zero;
      const auto& row_start = channel * channel_size;
      const auto& row_end = (channel+1) * channel_size;
      for (El::Int col = 0; col < local_width; ++col) {
        for (El::Int row = row_start; row < row_end; ++row) {
          const auto& x = local_input(row, col);
          sum += x;
          sqsum += x * x;
        }
      }
      local_mean(channel, 0) = sum;
      local_var(channel, 0) = sqsum;
    }
    El::Int num_per_sum;
    if (l.m_statistics_group_size == 0) {
      // Global statistics aggregation; allreduce on fused buffer.
      l.m_comm->allreduce(*l.m_mean_and_var, l.m_mean_and_var->RedundantComm(),
                        El::mpi::SUM);
      num_per_sum = channel_size * width;
    } else if (l.m_statistics_group_size == 1) {
      // Local aggregation, no allreduce needed.
      num_per_sum = channel_size * local_width;
    } else {
      // Grouped batchnorm. Allreduce on fused buffer.
      l.m_comm->allreduce(*l.m_mean_and_var,
                        l.m_comm->get_packed_group_comm(l.m_statistics_group_size),
                        El::mpi::SUM);
      if (l.m_num_per_sum_cache.count(width) == 0) {
        num_per_sum = channel_size * local_width;
        num_per_sum = l.m_comm->allreduce(
          num_per_sum, l.m_comm->get_packed_group_comm(l.m_statistics_group_size));
        l.m_num_per_sum_cache[width] = num_per_sum;
      } else {
        num_per_sum = l.m_num_per_sum_cache[width];
      }
    }

    // Compute minibatch statistics
    if (num_per_sum <= 1) {
      El::Fill(local_var, one);
    } else {
      LBANN_OMP_PARALLEL_FOR
      for (El::Int channel = 0; channel < num_channels; ++channel) {
        const auto& mean = local_mean(channel, 0) / num_per_sum;
        const auto& sqmean = local_var(channel, 0) / num_per_sum;
        auto var = num_per_sum * (sqmean - mean * mean) / (num_per_sum - 1);
        var = std::max(var, l.m_epsilon);
        local_mean(channel, 0) = mean;
        local_var(channel, 0) = var;
        auto& running_mean = local_running_mean(channel, 0);
        auto& running_var = local_running_var(channel, 0);
        running_mean = l.m_decay * running_mean + (one - l.m_decay) * mean;
        running_var = l.m_decay * running_var + (one - l.m_decay) * var;
      }
    }

  }

  // Get matrices
  const auto& local_scale = l.m_weights[0]->get_values().LockedMatrix();
  const auto& local_bias = l.m_weights[1]->get_values().LockedMatrix();
  const auto& local_mean = (is_training ?
                            l.m_mean_v->LockedMatrix() :
                            l.m_weights[2]->get_values().LockedMatrix());
  const auto& local_var = (is_training ?
                           l.m_var_v->LockedMatrix() :
                           l.m_weights[3]->get_values().LockedMatrix());

  // Iterate through channels
  LBANN_OMP_PARALLEL_FOR
  for (El::Int channel = 0; channel < num_channels; ++channel) {

    // Get channel parameters
    const auto& mean = local_mean(channel, 0);
    const auto& var = local_var(channel, 0);
    const TensorDataType inv_stdev = 1 / std::sqrt(var + l.m_epsilon);
    const auto& scale = local_scale(channel, 0);
    const auto& bias = local_bias(channel, 0);

    // Apply batch normalization to inputs in channel
    const auto& row_start = channel * channel_size;
    const auto& row_end = (channel+1) * channel_size;
    for (El::Int col = 0; col < local_width; ++col) {
      for (El::Int row = row_start; row < row_end; ++row) {
        const auto& x = local_input(row, col);
        const auto& xhat = (x - mean) * inv_stdev;
        auto& y = local_output(row, col);
        y = scale * xhat + bias;
      }
    }

  }

}

template <typename TensorDataType>
void bp_compute_impl(batch_normalization_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>& l) {
  constexpr TensorDataType one = 1;
  const bool is_training = l.m_model->get_execution_context().get_execution_mode() == execution_mode::training;

  // Matrices
  const auto& local_scale = l.m_weights[0]->get_values().LockedMatrix();
  const auto& local_mean = (is_training ?
                            l.m_mean_v->LockedMatrix() :
                            l.m_weights[2]->get_values().LockedMatrix());
  const auto& local_var = (is_training ?
                           l.m_var_v->LockedMatrix() :
                           l.m_weights[3]->get_values().LockedMatrix());
  const auto& input = l.get_prev_activations();
  const auto& local_input = input.LockedMatrix();
  const auto& local_gradient_wrt_output = l.get_local_prev_error_signals();
  auto& local_gradient_wrt_input = l.get_local_error_signals();
  auto& local_mean_gradient = l.m_mean_gradient_v->Matrix();
  auto& local_var_gradient = l.m_var_gradient_v->Matrix();
  auto& local_scale_gradient = l.m_scale_gradient->Matrix();
  auto& local_bias_gradient = l.m_bias_gradient->Matrix();

  // Matrix parameters
  const auto& c = static_cast<sgd_execution_context&>(l.m_model->get_execution_context());
  const auto effective_mini_batch_size = c.get_effective_mini_batch_size();
  const auto& width = input.Width();
  const auto& local_width = local_input.Width();
  const auto& output_dims = l.get_output_dims();
  const auto& num_channels = output_dims[0];
  const auto& channel_size = l.get_output_size() / num_channels;

  // Compute local gradients
  LBANN_OMP_PARALLEL_FOR
  for (El::Int channel = 0; channel < num_channels; ++channel) {

    // Initialize channel parameters and gradients
    const auto& mean = local_mean(channel, 0);
    const auto& var = local_var(channel, 0);
    const auto& scale = local_scale(channel, 0);
    const TensorDataType inv_stdev = 1 / std::sqrt(var + l.m_epsilon);
    const auto& dvar_factor = inv_stdev * inv_stdev * inv_stdev / 2;
    TensorDataType dmean = 0;
    TensorDataType dvar = 0;
    TensorDataType dscale = 0;
    TensorDataType dbias = 0;

    // Compute gradient contributions from local entries
    const auto& row_start = channel * channel_size;
    const auto& row_end = (channel+1) * channel_size;
    for (El::Int col = 0; col < local_width; ++col) {
      for (El::Int row = row_start; row < row_end; ++row) {
        const auto& x = local_input(row, col);
        const auto& xhat = (x - mean) * inv_stdev;
        const auto& dy = local_gradient_wrt_output(row, col);
        dscale += dy * xhat;
        dbias += dy;
        const auto& dxhat = dy * scale;
        dmean += - dxhat * inv_stdev;
        dvar += - dxhat * (x - mean) * dvar_factor;
      }
    }
    local_mean_gradient(channel, 0) = dmean;
    local_var_gradient(channel, 0) = dvar;
    local_scale_gradient(channel, 0) = dscale;
    local_bias_gradient(channel, 0) = dbias;

  }

  // Accumulate gradients
  if (is_training) {
    if (l.m_statistics_group_size == 0) {
      // Global aggregation; allreduce on fused buffer.
      l.m_comm->allreduce(*l.m_mean_and_var_gradient,
                        l.m_mean_and_var_gradient->RedundantComm(),
                        El::mpi::SUM);
    } else if (l.m_statistics_group_size > 1) {
      // Grouped batchnorm; allreduce on fused buffer.
      l.m_comm->allreduce(*l.m_mean_and_var_gradient,
                        l.m_comm->get_packed_group_comm(l.m_statistics_group_size),
                        El::mpi::SUM);
    }
  } else {
    // Zero fused buffer.
    El::Zero(*l.m_mean_and_var_gradient);
  }
  optimizer<TensorDataType>* scale_optimizer = l.m_weights[0]->get_optimizer();
  if (scale_optimizer != nullptr) {
    scale_optimizer->add_to_gradient(*l.m_scale_gradient,
                                     one / effective_mini_batch_size,
                                     true);
  }
  optimizer<TensorDataType>* bias_optimizer = l.m_weights[1]->get_optimizer();
  if (bias_optimizer != nullptr) {
    bias_optimizer->add_to_gradient(*l.m_bias_gradient,
                                    one / effective_mini_batch_size,
                                    true);
  }

  // Compute error signal
  El::Int num_per_sum;
  if (l.m_statistics_group_size == 0) {
    // Global statistics aggregation.
    num_per_sum = channel_size * width;
  } else if (l.m_statistics_group_size == 1) {
    // Local aggregation.
    num_per_sum = channel_size * local_width;
  } else {
    // Grouped batchnorm.
    num_per_sum = l.m_num_per_sum_cache[width];  // This was computed in FP.
  }
  if (num_per_sum <= 1) {
    El::Zero(local_gradient_wrt_input);
  } else {
    LBANN_OMP_PARALLEL_FOR
    for (El::Int channel = 0; channel < num_channels; ++channel) {

      // Initialize channel parameters and gradients
      const auto& mean = local_mean(channel, 0);
      const auto& var = local_var(channel, 0);
      const auto& scale = local_scale(channel, 0);
      const auto& dmean = local_mean_gradient(channel, 0);
      const auto& dvar = local_var_gradient(channel, 0);

      // Compute useful constants
      const TensorDataType inv_stdev = 1 / std::sqrt(var + l.m_epsilon);
      const auto& dmean_term = dmean / num_per_sum;
      const auto& dvar_term = dvar * 2 / (num_per_sum - 1);

      // Compute error signal for current channel
      const auto& row_start = channel * channel_size;
      const auto& row_end = (channel+1) * channel_size;
      for (El::Int col = 0; col < local_width; ++col) {
        for (El::Int row = row_start; row < row_end; ++row) {
          const auto& x = local_input(row, col);
          const auto& dy = local_gradient_wrt_output(row, col);
          const auto& dxhat = dy * scale;
          auto& dx = local_gradient_wrt_input(row, col);
          dx = dxhat * inv_stdev + dmean_term + dvar_term * (x - mean);
        }
      }

    }
  }

}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void batch_normalization_layer<TensorDataType, T_layout, Dev>::fp_compute() {
  fp_compute_impl<TensorDataType>(*this);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void batch_normalization_layer<TensorDataType, T_layout, Dev>::bp_compute() {
  bp_compute_impl<TensorDataType>(*this);
}

template class batch_normalization_layer<
  float, data_layout::DATA_PARALLEL, El::Device::CPU>;

} // namespace lbann

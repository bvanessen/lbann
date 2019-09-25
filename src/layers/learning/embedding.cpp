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

#define LBANN_EMBEDDING_LAYER_INSTANTIATE
#include "lbann/layers/learning/embedding.hpp"
#include "lbann/models/model.hpp"
#include "lbann/execution_contexts/sgd_execution_context.hpp"

namespace lbann {

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void embedding_layer<TensorDataType, T_layout, Dev>::setup_matrices(const El::Grid& grid) {
  data_type_layer<TensorDataType>::setup_matrices(grid);
  if(Dev == El::Device::CPU) {
    if(T_layout == data_layout::DATA_PARALLEL) {
      // Allocate a StarMat
      this->m_dictionary_gradient = El::DistMatrix<TensorDataType, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU>(grid);
    }
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void embedding_layer<TensorDataType, T_layout, Dev>::setup_dims() {
  data_type_layer<TensorDataType>::setup_dims();

  if(Dev == El::Device::CPU) {
    if(T_layout == data_layout::DATA_PARALLEL) {
      // Make sure input dimensions are valid
      if (this->get_input_size() != 1) {
        const auto& input_dims = this->get_input_dims();
        std::ostringstream err;
        err << get_type() << " layer \"" << this->get_name() << "\" "
            << "recieved an input tensor with invalid dimensions "
            << "(expected 1, got ";
        for (size_t i = 0; i < input_dims.size(); ++i) {
          err << (i > 0 ? "x" : "") << input_dims[i];
        }
        err << ")";
        LBANN_ERROR(err.str());
      }

      // Output is size of embedding vector
      this->set_output_dims({static_cast<int>(m_embedding_size)});
    }
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void embedding_layer<TensorDataType, T_layout, Dev>::setup_data() {
  data_type_layer<TensorDataType>::setup_data();

  if(Dev == El::Device::CPU) {
    if(T_layout == data_layout::DATA_PARALLEL) {
      // Make sure layer has weights for dictionary
      if (this->m_weights.size() != 1) {
        std::ostringstream err;
        err << "attempted to setup "
            << this->get_type() << " layer \"" << this->get_name() << "\" "
            << "with an invalid number of weights "
            << "(expected 1, "
            << "found " << this->m_weights.size() << ")";
        LBANN_ERROR(err.str());
      }

      // Initialize dictionary
      auto& dict = *this->get_weights()[0];
      auto matrix_dist = this->get_prev_activations().DistData();
      matrix_dist.colDist = El::STAR;
      matrix_dist.rowDist = El::STAR;
      dict.set_dims({static_cast<int>(m_embedding_size)},
                    {static_cast<int>(m_dictionary_size)});
      dict.set_matrix_distribution(matrix_dist);

      // Initialize gradient w.r.t. dictionary
      m_dictionary_gradient.Resize(m_embedding_size, m_dictionary_size);
    }
  }
}

template <typename TensorDataType>
void fp_compute_impl(embedding_layer<TensorDataType, data_layout::DATA_PARALLEL,El::Device::CPU>& l) {

  // Local data
  const auto& local_dict = l.get_weights()[0]->get_values().LockedMatrix();
  const auto& local_input = l.get_local_prev_activations();
  auto& local_output = l.get_local_activations();
  const auto& local_width = local_input.Width();

  // Populate output matrix with appropriate columns of dictionary
  El::Matrix<TensorDataType, El::Device::CPU> dict_v, output_v;
  for (El::Int col = 0; col < local_width; ++ col) {
    const El::Int ind = static_cast<El::Int>(local_input(0, col));
    El::LockedView(dict_v, local_dict, El::ALL, El::IR(ind));
    El::View(output_v, local_output, El::ALL, El::IR(col));
    El::Copy(dict_v, output_v);
  }

}

template <typename TensorDataType>
void bp_compute_impl(embedding_layer<TensorDataType, data_layout::DATA_PARALLEL,El::Device::CPU>& l) {

  // Embedding layer is not differentiable w.r.t. inputs
  El::Zero(l.get_error_signals());

  // Nothing to be done if dictionary is not being optimized
  if (l.get_weights()[0]->get_optimizer() == nullptr) { return; }
  auto& opt = *l.get_weights()[0]->get_optimizer();

  // Local data
  const auto& local_input = l.get_local_prev_activations();
  auto& local_dict_grad = l.m_dictionary_gradient.Matrix();
  const auto& local_output_grad = l.get_local_prev_error_signals();
  const auto& local_width = local_input.Width();
  const auto& c = static_cast<const sgd_execution_context&>(l.m_model->get_execution_context());
  const auto& mini_batch_size = c.get_effective_mini_batch_size();

  // Update appropriate columns of gradient w.r.t. dictionary
  El::Zero(local_dict_grad);
  El::Matrix<TensorDataType, El::Device::CPU> dict_grad_v, output_grad_v;
  for (El::Int col = 0; col < local_width; ++ col) {
    const El::Int ind = static_cast<El::Int>(local_input(0, col));
    El::View(dict_grad_v, local_dict_grad, El::ALL, El::IR(ind));
    El::LockedView(output_grad_v, local_output_grad, El::ALL, El::IR(col));
    El::Axpy(DataType{1}, output_grad_v, dict_grad_v);
  }
  opt.add_to_gradient(l.m_dictionary_gradient,
                      TensorDataType{1} / mini_batch_size,
                      true);

}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void embedding_layer<TensorDataType, T_layout, Dev>::fp_compute() {
  fp_compute_impl<TensorDataType>(*this);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void embedding_layer<TensorDataType, T_layout, Dev>::bp_compute() {
  bp_compute_impl<TensorDataType>(*this);
}

// Explicit instantiation
template class embedding_layer<float, data_layout::DATA_PARALLEL, El::Device::CPU>;
//template class embedding_layer<double, data_layout::DATA_PARALLEL, El::Device::CPU>;

} // namespace lbann

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

#ifndef LBANN_LAYERS_LEARNING_EMBEDDING_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_EMBEDDING_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/memory.hpp"

namespace lbann {

/** @brief Lookup table to vectors of fixed size.
 *
 *  Takes a scalar input, interprets it as an index, and outputs the
 *  corresponding vector. The number of embedding vectors and the size
 *  of vectors are fixed. If the index is out-of-range, then the
 *  output is a vector of zeros.
 *
 *  The embedding vectors are stored in an
 *  @f$ \text{embedding\_dim} \times \text{num\_embeddings} @f$
 *  weights matrix. Note that this is the transpose of the weights in
 *  the PyTorch embedding layer.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class embedding_layer : public data_type_layer<TensorDataType> {
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "embedding layer only supports data parallel layout");
  static_assert(Device == El::Device::CPU,
                "embedding layer only supports CPU");
public:

  /**
   *  @param comm           LBANN communicator.
   *  @param num_embeddings Size of dictionary of embeddings.
   *  @param embedding_dim  Size of embedding vectors.
   *  @param padding_idx    If set, then the corresponding embedding
   *                        vector is initialized with zeros. The
   *                        objective function gradient w.r.t. this
   *                        embedding vector is always zero.
   */
  embedding_layer(lbann_comm* comm,
                  size_t num_embeddings,
                  size_t embedding_dim,
                  El::Int padding_idx=-1)
    : data_type_layer<TensorDataType>(comm),
      m_num_embeddings{num_embeddings},
      m_embedding_dim{embedding_dim},
      m_padding_idx{padding_idx} {}

  embedding_layer(const embedding_layer& other) = default;
  embedding_layer& operator=(const embedding_layer& other) = default;
  ~embedding_layer() = default;

  embedding_layer* copy() const override {
    return new embedding_layer(*this);
  }

  std::string get_type() const override { return "embedding"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  description get_description() const override;

protected:

  void setup_matrices(const El::Grid& grid) override;
  void setup_dims() override;
  void setup_data() override;

  void fp_compute() override;
  void bp_compute() override;

private:

  /** Size of dictionary of embeddings. */
  size_t m_num_embeddings;
  /** Size of embedding vectors. */
  size_t m_embedding_dim;
  /** If the padding index is set, then the corresponding embedding
   *  vector is initialized with zeros. The objective function
   *  gradient w.r.t. this embedding vector is always zero.
   */
  El::Int m_padding_idx;

  /** Gradient w.r.t. embedding weights. */
  El::DistMatrix<TensorDataType, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU> m_dictionary_gradient;

  template <typename U>
  friend void fp_compute_impl(embedding_layer<U, Layout, Device>& l);
  template <typename U>
  friend void bp_compute_impl(embedding_layer<U, Layout, Device>& l);
};

// =========================================================
// Implementation
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
description embedding_layer<TensorDataType,Layout,Device>::get_description() const {
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Num embeddings", m_num_embeddings);
  desc.add("Embedding dim", m_embedding_dim);
  desc.add("Padding index", m_padding_idx);
  return desc;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void embedding_layer<TensorDataType,Layout,Device>::setup_dims() {
  data_type_layer<TensorDataType>::setup_dims();

  // Make sure input dimensions are valid
  if (this->get_input_size() != 1) {
    const auto& dims = this->get_input_dims();
    std::ostringstream dims_ss;
    for (size_t i = 0; i < dims.size(); ++i) {
      dims_ss << (i > 0 ? "x" : "") << dims[i];
    }
    LBANN_ERROR(this->get_type()," layer \"",this->get_name(),"\" ",
                "recieved an input tensor with invalid dimensions "
                "(expected 1, got ",dims_ss.str(),")");
  }

  // Output is size of embedding vector
  this->set_output_dims({static_cast<int>(m_embedding_dim)});

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void embedding_layer<TensorDataType, Layout,Device>::setup_data() {
  data_type_layer<TensorDataType>::setup_data();

  // Construct default weights if needed
  // Note: Randomly drawn from normal distribution with mean 0 and
  // standard deviation 1.
  if (this->m_weights.empty()) {
    auto w = make_unique<weights>(this->get_comm());
    auto init = make_unique<normal_initializer<TensorDataType>>(0,1);
    auto opt = std::unique_ptr<optimizer>(dynamic_cast<data_type_optimizer<TensorDataType>*>(this->m_model->create_optimizer()));
    w->set_name(this->get_name() + "_weights");
    w->set_initializer(std::move(init));
    w->set_optimizer(std::move(opt));
    this->m_weights.push_back(w.get());
    this->m_model->add_weights(std::move(w));
  }
  if (this->m_weights.size() != 1) {
    LBANN_ERROR("attempted to setup ",
                this->get_type()," layer \"",this->get_name(),"\" ",
                "with an invalid number of weights ",
                "(expected 1, found ",this->m_weights.size(),")");
  }

  // Initialize dictionary
  auto& dict = *this->m_weights[0];
  auto matrix_dist = this->get_prev_activations().DistData();
  matrix_dist.colDist = El::STAR;
  matrix_dist.rowDist = El::STAR;
  dict.set_dims({static_cast<int>(m_embedding_dim)},
                {static_cast<int>(m_num_embeddings)});
  dict.set_matrix_distribution(matrix_dist);
  dict.setup();

  // Zero out embedding vector for padding index
  if (0 <= m_padding_idx
      && m_padding_idx < static_cast<El::Int>(m_embedding_dim)) {
    auto& dict_values = dict.get_values();
    std::unique_ptr<AbsDistMat> pad_embedding(dict_values.Construct(dict_values.Grid(),
                                                                    dict_values.Root()));
    El::View(*pad_embedding, dict_values, El::ALL, El::IR(m_padding_idx));
    El::Zero(*pad_embedding);
  }

  // Initialize gradient w.r.t. dictionary
  m_dictionary_gradient.Resize(m_embedding_dim, m_num_embeddings);

}

#ifndef LBANN_EMBEDDING_LAYER_INSTANTIATE
extern template class embedding_layer<
  float, data_layout::DATA_PARALLEL, El::Device::CPU>;
#endif // LBANN_EMBEDDING_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_EMBEDDING_HPP_INCLUDED

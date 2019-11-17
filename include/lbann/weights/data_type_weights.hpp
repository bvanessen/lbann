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

#ifndef LBANN_DATA_TYPE_WEIGHTS_HPP
#define LBANN_DATA_TYPE_WEIGHTS_HPP

#include "lbann/weights/weights.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/optimizers/data_type_optimizer.hpp"

namespace lbann_data {
class WeightsData;
}

namespace lbann {

// Forward declaration
// template <typename TensorDataType>
// class data_type_optimizer;

/** Neural network weights.
 *  Weights are tensors that act as trainable parameters for a neural
 *  network. The values can be initialized with a weights initializer
 *  and are optimized with first-order methods (e.g. stochastic
 *  gradient descent).
 *
 *  Internally, the weight values are stored in a 2D distributed
 *  matrix. The "matrix height dimensions" are tensor dimensions that
 *  correspond to the matrix height. The remaining dimensions, the
 *  "matrix width dimensions," correspond to the matrix width.
 *
 *  Note that LBANN weights are similar to Tensorflow variables and
 *  Caffe parameters.
 */
template <typename TensorDataType>
class data_type_weights : public weights {
  friend class data_type_optimizer<TensorDataType>;

public:
  data_type_weights(lbann_comm* comm);
  data_type_weights(const data_type_weights& other);
  data_type_weights& operator=(const data_type_weights& other);
  virtual ~data_type_weights() = default;

  /** Create a copy of the weights.
   *  This function dynamically allocates memory for a weights
   *  instance and instantiates a copy. The caller is responsible for
   *  deallocating the instance.
   */
  data_type_weights* copy() const override { return new data_type_weights(*this); }

  /** Human-readable description. */
  description get_description() const override;

  bool has_optimizer() const override { return m_optimizer != nullptr; }

  // -----------------------------------------------
  // Dimension accessors
  // -----------------------------------------------
  void set_dims(std::vector<int> matrix_height_dims,
                std::vector<int> matrix_width_dims = std::vector<int>()) override;
  /** Set weight tensor dimensions as a 1D tensor. */
  void set_dims(int size) override { set_dims({size}, {}); }


  // -----------------------------------------------
  // Initializer accessors
  // -----------------------------------------------
  /** Get weights initializer. */
  data_type_weights_initializer<TensorDataType>* get_initializer() override;
  /** Get weights initializer (const). */
  const data_type_weights_initializer<TensorDataType>* get_initializer() const override;
  /** Set weights initializer.
   *  The contents of 'init' are moved to a class member.
   */
  void set_initializer(std::unique_ptr<weights_initializer>&& init) override;

  // -----------------------------------------------
  // Optimizer accessors
  // -----------------------------------------------
  /** Get weights optimizer.
   *  Returns a null pointer if the weights are frozen.
   */
  data_type_optimizer<TensorDataType>* get_optimizer() override;
  /** Get weights optimizer.
   *  Returns a null pointer if the weights are frozen.
   */
  const data_type_optimizer<TensorDataType>* get_optimizer() const override;
  /** Set weights optimizer.
   *  The contents of opt are moved to a class member.
   */
  void set_optimizer(std::unique_ptr<optimizer>&& opt) override;

  // -----------------------------------------------
  // Setup
  // -----------------------------------------------
  void setup() override;

  // -----------------------------------------------
  // Weight matrix accessors
  // -----------------------------------------------

  /** Get the weight matrix. */
  El::AbstractDistMatrix<TensorDataType>& get_values();
  /** Get the weight matrix. */
  const El::AbstractDistMatrix<TensorDataType>& get_values() const;
  /** Set the weight matrix. */
  void set_values(const El::AbstractDistMatrix<TensorDataType>& values);

  /** Set a weight value. */
  void set_value(TensorDataType value, int index);
  /** Set an entry in the weight tensor. */
  void set_value(TensorDataType value, std::vector<int> pos);
  /** Set an entry in the weight matrix. */
  void set_value(TensorDataType value, int row, int col);

  /** Reconcile weight values.
   *  If weight values are duplicated across multiple processes, they
   *  are set to the average across the processes.
   */
  void reconcile_values();
  /** Asynchronously reconcile weight values.
   *  If weight values are duplicated across multiple processes, they
   *  are set to the average across the processes.
   */
  void reconcile_values(Al::request& req);

  // -----------------------------------------------
  // Checkpointing
  // -----------------------------------------------
  bool save_to_checkpoint_shared(persist& p) override;
  bool load_from_checkpoint_shared(persist& p) override;
  bool load_from_save(std::string const& ckpt_dir, std::vector<std::string> const& weight_list) override;
  bool save_to_checkpoint_distributed(persist& p) override;
  bool load_from_checkpoint_distributed(persist& p) override;

  /** Write weights to proto file */
  void write_proto(lbann_data::WeightsData* proto) const override;

private:

  /** Weight matrix. */
  std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> m_values;

  /** Weights initializer.
   *  Default is nullptr, which corresponds to zero initialization.
   */
  std::unique_ptr<data_type_weights_initializer<TensorDataType>> m_initializer;
  /** Weights optimizer.
   *  Default is nullptr, which corresponds to no optimizer.
   */
  std::unique_ptr<data_type_optimizer<TensorDataType>> m_optimizer;

};

} // namespace lbann

#endif // LBANN_DATA_TYPE_WEIGHTS_HPP

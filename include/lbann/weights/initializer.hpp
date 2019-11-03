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

#ifndef LBANN_WEIGHTS_INITIALIZER_HPP
#define LBANN_WEIGHTS_INITIALIZER_HPP

#include "lbann/base.hpp"
#include "lbann/utils/description.hpp"

#include <google/protobuf/message.h>

namespace lbann {

/** @brief Scheme for initializing weight values. */
template <typename TensorDataType>
class weights_initializer {
public:
  weights_initializer() = default;
  virtual ~weights_initializer() = default;

  /** Human-readable string describing concrete class. */
  virtual std::string get_type() const = 0;

  /** Human-readable description of class instance. */
  virtual description get_description() const;

  /** Create a copy. */
  virtual weights_initializer* copy() const = 0;

  /** Initialize entries in a weights matrix. */
  virtual void fill(El::AbstractDistMatrix<TensorDataType>& matrix) = 0;

};

/** @brief Fill weights with a constant value. */
template <typename TensorDataType>
class constant_initializer : public weights_initializer<TensorDataType> {
public:
  constant_initializer(TensorDataType value)
    : weights_initializer<TensorDataType>(), m_value(value) {}
  constant_initializer* copy() const override {
    return new constant_initializer(*this);
  }
  std::string get_type() const override { return "constant"; }
  description get_description() const override;
  void fill(El::AbstractDistMatrix<TensorDataType>& matrix) override;

private:

  /** Weights value. */
  TensorDataType m_value;

};

/** @brief Fill weights with values from a list.
 *
 *  The number of weight entries must exactly match the number of
 *  provided values.
 */
template <typename TensorDataType>
class value_initializer : public weights_initializer<TensorDataType> {
public:
  value_initializer(std::vector<TensorDataType> values)
    : weights_initializer<TensorDataType>(), m_values(std::move(values)) {}
  value_initializer* copy() const override {
    return new value_initializer(*this);
  }
  std::string get_type() const override { return "value"; }
  void fill(El::AbstractDistMatrix<TensorDataType>& matrix) override;

private:

  /** List of weights values. */
  std::vector<TensorDataType> m_values;

};

/** @brief Draw weights values from a uniform random distribution. */
template <typename TensorDataType>
class uniform_initializer : public weights_initializer<TensorDataType> {
 public:
  uniform_initializer(TensorDataType min = TensorDataType(0),
                      TensorDataType max = TensorDataType(1))
    : weights_initializer<TensorDataType>(), m_min(min), m_max(max) {}
  uniform_initializer* copy() const override {
    return new uniform_initializer(*this);
  }
  std::string get_type() const override{ return "uniform"; }
  description get_description() const override;
  void fill(El::AbstractDistMatrix<TensorDataType>& matrix) override;

private:

  /** Uniform distribution minimum. */
  TensorDataType m_min;
  /** Uniform distribution maximum. */
  TensorDataType m_max;

};

/** @brief Draw weights values from a normal random distribution. */
template <typename TensorDataType>
class normal_initializer : public weights_initializer<TensorDataType> {
public:
  normal_initializer(TensorDataType mean = TensorDataType(0),
                     TensorDataType standard_deviation = TensorDataType(1))
    : weights_initializer<TensorDataType>(),
      m_mean(mean),
      m_standard_deviation(standard_deviation) {}
  normal_initializer* copy() const override {
    return new normal_initializer(*this);
  }
  std::string get_type() const override { return "normal"; }
  description get_description() const override;
  void fill(El::AbstractDistMatrix<TensorDataType>& matrix) override;

private:

  /** Normal distribution mean. */
  TensorDataType m_mean;
  /** Normal distribution standard deviation. */
  TensorDataType m_standard_deviation;

};

template <typename TensorDataType>
std::unique_ptr<weights_initializer<TensorDataType>>
build_constant_initializer_from_pbuf(google::protobuf::Message const& msg);
template <typename TensorDataType>
std::unique_ptr<weights_initializer<TensorDataType>>
build_value_initializer_from_pbuf(google::protobuf::Message const& msg);
template <typename TensorDataType>
std::unique_ptr<weights_initializer<TensorDataType>>
build_uniform_initializer_from_pbuf(google::protobuf::Message const& msg);
template <typename TensorDataType>
std::unique_ptr<weights_initializer<TensorDataType>>
build_normal_initializer_from_pbuf(google::protobuf::Message const& msg);

} // namespace lbann

#endif // LBANN_WEIGHTS_INITIALIZER_HPP

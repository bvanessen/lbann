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

#include "lbann/optimizers/optimizer.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

std::string to_string(optimizer_gradient_status status) {
  switch (status) {
  case optimizer_gradient_status::ready:
    return "ready";
  case optimizer_gradient_status::cleared:
    return "cleared";
  case optimizer_gradient_status::allreduce_needed:
    return "allreduce needed";
  case optimizer_gradient_status::allreduce_started:
    return "allreduce started";
  default:
    return "unknown";
  }
}

template <typename TensorDataType>
optimizer<TensorDataType>::optimizer(TensorDataType learning_rate)
  : m_comm(nullptr), m_learning_rate(learning_rate) {}

template <typename TensorDataType>
optimizer<TensorDataType>::optimizer(const optimizer<TensorDataType>& other)
  : m_comm(other.m_comm),
    m_weights(other.m_weights),
    m_gradient(other.m_gradient ? other.m_gradient->Copy() : nullptr),
    m_gradient_v(other.m_gradient_v ? other.m_gradient_v->Copy() : nullptr),
    m_gradient_sources(other.m_gradient_sources),
    m_gradient_status(other.m_gradient_status),
    m_learning_rate(other.m_learning_rate),
    m_step_time(other.m_step_time) {
  if (m_gradient_status == optimizer_gradient_status::allreduce_started) {
    LBANN_ERROR("attempted to copy optimizer while a "
                "gradient allreduce is in progress");
  }
}

template <typename TensorDataType>
optimizer<TensorDataType>& optimizer<TensorDataType>::operator=(const optimizer<TensorDataType>& other) {
  m_comm = other.m_comm;
  m_weights = other.m_weights;
  m_gradient.reset(other.m_gradient ? other.m_gradient->Copy() : nullptr);
  m_gradient_v.reset(other.m_gradient_v ? other.m_gradient_v->Copy() : nullptr);
  m_gradient_sources = other.m_gradient_sources;
  m_gradient_status = other.m_gradient_status;
  m_learning_rate = other.m_learning_rate;
  m_step_time = other.m_step_time;
  if (m_gradient_status == optimizer_gradient_status::allreduce_started) {
    LBANN_ERROR("attempted to copy optimizer while a "
                "gradient allreduce is in progress");
  }
  return *this;
}

template <typename TensorDataType>
description optimizer<TensorDataType>::get_description() const {
  description desc(get_type() + " optimizer");
  desc.add("Learning rate", m_learning_rate);
  return desc;
}

template <typename TensorDataType>
data_type_weights<TensorDataType>& optimizer<TensorDataType>::get_weights() {
  // Item 3, p. 23 in "Effective C++", 3rd ed., by Scott Meyers
  return const_cast<data_type_weights<TensorDataType>&>(static_cast<const optimizer&>(*this).get_weights());
}

template <typename TensorDataType>
const data_type_weights<TensorDataType>& optimizer<TensorDataType>::get_weights() const {
  if (m_weights == nullptr) {
    LBANN_ERROR("attempted to access the weights being optimized "
                "before they are set");
  }
  return *m_weights;
}

template <typename TensorDataType>
El::AbstractDistMatrix<TensorDataType>& optimizer<TensorDataType>::get_gradient() {

  // Make sure gradient matrix has been setup
  if (m_gradient == nullptr) {
    LBANN_ERROR("attempted to access gradient before it is set up");
  }

  // Make sure gradient values are ready
  start_gradient_allreduce();
  finish_gradient_allreduce();
  if (m_gradient_status == optimizer_gradient_status::cleared) {
    El::Zero(*m_gradient);
    m_gradient_status = optimizer_gradient_status::ready;
  }
  if (m_gradient_status != optimizer_gradient_status::ready) {
    std::ostringstream err;
    err << "unexpected gradient status (expected "
        << "\"" << to_string(optimizer_gradient_status::ready) << "\", "
        << "but found \"" << to_string(m_gradient_status) << "\")";
    LBANN_ERROR(err.str());
  }

  // Return gradient
  return *m_gradient;

}

template <typename TensorDataType>
void optimizer<TensorDataType>::add_to_gradient(const El::AbstractDistMatrix<TensorDataType>& gradient,
                                TensorDataType scale,
                                bool allreduce_needed) {

  // Check that matrices have been setup
  if (m_gradient == nullptr || m_gradient_v == nullptr) {
    LBANN_ERROR("attempted to access gradient before it is set up");
  }
  if (scale == DataType(0)) { return; }

  // Make sure input matrix is in correct distribution
  // Note: If input matrix is already in correct distribution, just
  // make a matrix view. Otherwise redistribute and possibly allreduce
  // the matrix.
  m_gradient_v->Empty();
  m_gradient_v->AlignWith(*m_gradient);
  if (m_gradient_v->DistData() == gradient.DistData()) {
    El::LockedView(*m_gradient_v, gradient);
  } else if (allreduce_needed) {
    std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> temp(gradient.Copy());
    get_comm().allreduce(*temp, temp->RedundantComm());
    El::Copy(*temp, *m_gradient_v);
    allreduce_needed = false;
  } else {
    El::Copy(gradient, *m_gradient_v);
  }

  // Add to gradient
  if (m_gradient_status == optimizer_gradient_status::allreduce_started) {
    finish_gradient_allreduce();
  }
  switch (m_gradient_status) {
  case optimizer_gradient_status::ready:
    if (allreduce_needed) {
      // Properly scale contributions that have already been allreduced or that
      // do not need allreduces.
      El::Scale(DataType(1) / m_gradient->RedundantSize(), *m_gradient);
      m_gradient_status = optimizer_gradient_status::allreduce_needed;
    }
    El::Axpy(scale, *m_gradient_v, *m_gradient);
    break;
  case optimizer_gradient_status::cleared:
    El::Copy(*m_gradient_v, *m_gradient);
    El::Scale(scale, *m_gradient);
    m_gradient_status = (allreduce_needed ?
                         optimizer_gradient_status::allreduce_needed :
                         optimizer_gradient_status::ready);
    break;
  case optimizer_gradient_status::allreduce_needed:
    {
      // Properly scale data that does not need to be allreduced.
      const auto& scale_ = (allreduce_needed ?
                            scale :
                            scale / m_gradient->RedundantSize());
      El::Axpy(scale_, *m_gradient_v, *m_gradient);
    }
    break;
  case optimizer_gradient_status::allreduce_started:
  default:
    LBANN_ERROR("unexpected gradient status "
                "(" + to_string(m_gradient_status) + ")");
  }

  // Clean up
  m_gradient_v->Empty();

}

template <typename TensorDataType>
void optimizer<TensorDataType>::clear_gradient() {
  if (m_gradient_status == optimizer_gradient_status::allreduce_started) {
    finish_gradient_allreduce();
  }
  m_gradient_status = optimizer_gradient_status::cleared;
  m_gradient_sources.clear();
}

template <typename TensorDataType>
El::AbstractDistMatrix<TensorDataType>& optimizer<TensorDataType>::get_gradient_buffer(TensorDataType& buf_scale,
                                           TensorDataType& in_scale,
                                           bool allreduce_needed) {
  if (m_gradient == nullptr) {
    LBANN_ERROR("attempted to access gradient before it is set up");
  }

  // Complete outstanding allreduce.
  if (m_gradient_status == optimizer_gradient_status::allreduce_started) {
    finish_gradient_allreduce();
  }
  // Determine scaling factor and transition state.
  switch (m_gradient_status) {
  case optimizer_gradient_status::ready:
    buf_scale = DataType(1);
    in_scale = DataType(1);
    if (allreduce_needed) {
      buf_scale /= m_gradient->RedundantSize();
      m_gradient_status = optimizer_gradient_status::allreduce_needed;
    }
    break;
  case optimizer_gradient_status::cleared:
    buf_scale = DataType(0);
    in_scale = DataType(1);
    m_gradient_status = (allreduce_needed ?
                         optimizer_gradient_status::allreduce_needed :
                         optimizer_gradient_status::ready);
    break;
  case optimizer_gradient_status::allreduce_needed:
    buf_scale = DataType(1);
    // Properly scale data that does not need to be allreduced.
    in_scale = (allreduce_needed ?
                DataType(1) :
                DataType(1) / m_gradient->RedundantSize());
    break;
  case optimizer_gradient_status::allreduce_started:
  default:
    LBANN_ERROR("unexpected gradient status ("
                + to_string(m_gradient_status) + ")");
  }
  return *m_gradient;
}

template <typename TensorDataType>
El::Int optimizer<TensorDataType>::get_num_gradient_sources() const {
  return m_gradient_sources.size();
}

template <typename TensorDataType>
void optimizer<TensorDataType>::add_gradient_source(const void* source) {
  if (source != nullptr) {
    m_gradient_sources.insert(source);
  }
}

template <typename TensorDataType>
void optimizer<TensorDataType>::remove_gradient_source(const void* source) {
  m_gradient_sources.erase(nullptr);
  m_gradient_sources.erase(source);
  if (m_gradient_sources.empty()) {
    start_gradient_allreduce();
  }
}

template <typename TensorDataType>
void optimizer<TensorDataType>::setup(data_type_weights<TensorDataType>* w) {
  m_comm = &w->get_comm();
  clear_gradient();

  // Set weights being optimized
  if (w != nullptr) { set_weights(w); }
  if (m_weights == nullptr) {
    LBANN_ERROR("attempted to setup optimizer without weights");
  }

  // Initialize matrices
  const auto& height = m_weights->get_matrix_height();
  const auto& width = m_weights->get_matrix_width();
  const El::AbstractDistMatrix<TensorDataType>& values = m_weights->get_values();
  m_gradient.reset(El::AbstractDistMatrix<TensorDataType>::Instantiate(values.DistData()));
  m_gradient->AlignWith(values);
  m_gradient->Resize(height, width);
  m_gradient_v.reset(El::AbstractDistMatrix<TensorDataType>::Instantiate(values.DistData()));
  m_gradient_v->AlignWith(values);
#ifdef HYDROGEN_HAVE_CUB
  if (m_gradient_v->GetLocalDevice() == El::Device::GPU) {
    m_gradient_v->Matrix().SetMemoryMode(1); // CUB GPU memory pool
  }
#endif // HYDROGEN_HAVE_CUB

}

template <typename TensorDataType>
void optimizer<TensorDataType>::step() {
  if (m_weights == nullptr) {
    LBANN_ERROR("attempted to perform optimization step without weights");
  }
  const auto start_time = get_time();
  step_compute(m_weights->get_values(), get_gradient());
  m_step_time += get_time() - start_time;
}

template <typename TensorDataType>
TensorDataType optimizer<TensorDataType>::get_learning_rate() const {
  return m_learning_rate;
}

template <typename TensorDataType>
void optimizer<TensorDataType>::set_learning_rate(TensorDataType learning_rate) {
  m_learning_rate = learning_rate;
};

template <typename TensorDataType>
void optimizer<TensorDataType>::start_gradient_allreduce() {
  switch (m_gradient_status) {
  case optimizer_gradient_status::allreduce_needed:
    get_comm().nb_allreduce(*m_gradient,
                            m_gradient->RedundantComm(),
                            m_gradient_allreduce_req);
    m_gradient_status = optimizer_gradient_status::allreduce_started;
    break;
  case optimizer_gradient_status::ready:
  case optimizer_gradient_status::cleared:
  case optimizer_gradient_status::allreduce_started:
    break;
  default: LBANN_ERROR("unexpected gradient status "
                       "(" + to_string(m_gradient_status) + ")");
  }
}

template <typename TensorDataType>
void optimizer<TensorDataType>::finish_gradient_allreduce() {
  switch (m_gradient_status) {
  case optimizer_gradient_status::allreduce_started:
    get_comm().wait(m_gradient_allreduce_req);
    m_gradient_status = optimizer_gradient_status::ready;
    break;
  case optimizer_gradient_status::ready:
  case optimizer_gradient_status::cleared:
    break;
  case optimizer_gradient_status::allreduce_needed:
    LBANN_ERROR("attempted to finish gradient allreduce "
                "before starting it");
    break;
  default:
    LBANN_ERROR("unexpected gradient status "
                "(" + to_string(m_gradient_status) + ")");
  }
}

// =============================
// Checkpointing
// =============================

template <typename TensorDataType>
bool optimizer<TensorDataType>::save_to_checkpoint_shared(persist& p, std::string m_name) {
  //  m_learning_rate;
  p.write_datatype(persist_type::train, "learning_rate", m_learning_rate);
  return true;
}

template <typename TensorDataType>
bool optimizer<TensorDataType>::load_from_checkpoint_shared(persist& p, std::string m_name) {
  p.read_datatype(persist_type::train, "learning_rate", &m_learning_rate);
  get_comm().trainer_broadcast(0, m_learning_rate);
  return true;
}

template <typename TensorDataType>
bool optimizer<TensorDataType>::save_to_checkpoint_distributed(persist& p, std::string m_name) {
  p.write_datatype(persist_type::train, "learning_rate", m_learning_rate);
  return true;
}

template <typename TensorDataType>
bool optimizer<TensorDataType>::load_from_checkpoint_distributed(persist& p, std::string m_name) {
  p.read_datatype(persist_type::train, "learning_rate", &m_learning_rate);
  return true;
}

} // namespace lbann

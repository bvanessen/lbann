////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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

#define LBANN_DATA_TYPE_OPERATOR_INSTANTIATE
#include "lbann/operators/data_type_operator.hpp"

namespace lbann {

template <typename InputTensorDataType, typename OutputTensorDataType>
void DataTypeOperator<InputTensorDataType, OutputTensorDataType>::fp_compute(
  BaseDistMat const& input,
  BaseDistMat& output) const
{
  return fp_compute(dynamic_cast<InputAbsDistMatrixType const&>(input),
                    dynamic_cast<OutputAbsDistMatrixType&>(output));
}

template <typename InputTensorDataType, typename OutputTensorDataType>
void DataTypeOperator<InputTensorDataType, OutputTensorDataType>::bp_compute(
  BaseDistMat const& input,
  BaseDistMat const& gradient_wrt_output,
  BaseDistMat& gradient_wrt_input) const
{
  return bp_compute(
    dynamic_cast<InputAbsDistMatrixType const&>(input),
    dynamic_cast<OutputAbsDistMatrixType const&>(gradient_wrt_output),
    dynamic_cast<InputAbsDistMatrixType&>(gradient_wrt_input));
};

#define PROTO(T) template class DataTypeOperator<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann

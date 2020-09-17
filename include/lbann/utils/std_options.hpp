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


#ifndef LBANN_STD_OPTIONS_HPP
#define LBANN_STD_OPTIONS_HPP

namespace lbann {

#define MAX_RNG_SEEDS_DISPLAY "RNG seeds per trainer to display"
#define NUM_IO_THREADS "Num. IO threads"
#define DATA_STORE_FAIL_ON_MISSING_SAMPLES "Data store fail on missing samples"
#define SAMPLE_LIST_FAIL_ON_MISSING_FILES "Sample list fail on missing files"
#define SAMPLE_LIST_FAIL_ON_UNREADABLE_FILES "Sample list fail on unreadable files"

void construct_std_options();

} // namespace lbann

#endif // LBANN_STD_OPTIONS_HPP
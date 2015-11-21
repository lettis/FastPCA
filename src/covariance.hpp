#pragma once
/*
Copyright (c) 2014, Florian Sittel (www.lettis.net)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include "matrix.hpp"

namespace FastPCA {
  namespace {
    struct _CovAccumulation {
      CovAccumulation(SymmetricMatrix<double> m, std::vector<double> sum, std::size_t n);
      CovAccumulation(std::size_t n_observations, std::size_t n_observables);
      SymmetricMatrix<double> m;
      std::vector<double> sum_observations;
      std::size_t n_observations;
    };

    _CovAccumulation
    _accumulate_covariance(const Matrix<double>& m);

    _CovAccumulation
    _join_accumulations(const CovAccumulation& c1, const CovAccumulation& c2);

    SymmetricMatrix<double>
    _get_covariance(const CovAccumulation& acc);
  } // end local namespace

  SymmetricMatrix<double>
  covariance_matrix(const std::string filename
                  , const std::size_t max_chunk_size
                  , bool use_correlation);

  namespace Periodic {
    SymmetricMatrix<double>
    covariance_matrix(const std::string filename
                    , const std::size_t max_chunk_size
                    , bool use_correlation);
  }; // end namespace FastPCA::Periodic

} // end namespace FastPCA


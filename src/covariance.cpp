
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

#include "covariance.hpp"
#include "file_io.hpp"
#include "errors.hpp"
#include "util.hpp"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <cmath>

#include <omp.h>

extern "C" {
  // use xdrfile library to read/write xtc and trr files from gromacs
  #include "xdrfile/xdrfile.h"
  #include "xdrfile/xdrfile_xtc.h"
  #include "xdrfile/xdrfile_trr.h"
}


namespace FastPCA {

  namespace {

    _CovAccumulation::_CovAccumulation(SymmetricMatrix<double> m
                                     , std::size_t n)
      : m(m),
        n_observations(n) {
    }

    _CovAccumulation::_CovAccumulation(std::size_t n_observations
                                     , std::size_t n_observables)
      : m(SymmetricMatrix<double>(n_observables)),
        n_observations(n_observations) {
    }

    _CovAccumulation
    _accumulate_covariance(const Matrix<double>& m) {
      const std::size_t nr = m.n_rows();
      const std::size_t nc = m.n_cols();
      _CovAccumulation c(nr, nc);
      std::size_t i, j, t;
      double cc;
      // calculate (i,j)-part of covariances
      for (i = 0; i < nc; ++i) {
        for (j = 0; j <= i; ++j) {
          cc = 0.0;
          #pragma omp parallel for default(none)\
                                   firstprivate(i,j)\
                                   private(t)\
                                   shared(m,c)\
                                   reduction(+:cc)
          for (t = 0; t < nr; ++t) {
            cc += (m(t,i) * m(t,j));
          }
          c.m(i,j) = cc;
        }
      }
      c.n_observations = nr;
      return c;
    }

    _CovAccumulation
    _join_accumulations(const _CovAccumulation& c1
                      , const _CovAccumulation& c2) {
      assert(c1.m.n_cols() == c2.m.n_cols());
      return {c1.m+c2.m, c1.n_observations+c2.n_observations};
    }

    SymmetricMatrix<double>
    _get_covariance(const _CovAccumulation& acc) {
      std::size_t i, j;
      SymmetricMatrix<double> sm = acc.m;
      for (i = 0; i < sm.n_cols(); ++i) {
        for (j = 0; j <= i; ++j) {
          sm(i,j) /= (acc.n_observations-1);
        }
      }
      return sm;
    }
    
    SymmetricMatrix<double>
    _covariance_matrix(const std::string filename
                     , const std::size_t max_chunk_size
                     , bool use_correlation
                     , bool periodic
                     , Matrix<double> stats) {
      std::size_t n_variables = stats.n_rows();
      std::vector<double> means(n_variables);
      std::vector<double> shifts(n_variables);
      std::vector<double> inverse_sigmas(n_variables);
      for (std::size_t i=0; i < n_variables; ++i) {
        means[i] = stats(i,0);
        inverse_sigmas[i] = 1.0 / stats(i,1);
        if (periodic) {
          shifts[i] = stats(i,2);
        }
      }
      // take only half size because of intermediate results.
      std::size_t chunk_size = max_chunk_size / 2;
      // data files are often to big to read into RAM completely.
      // thus, we read the data blockwise, calculate the temporary
      // covariance data and accumulate the results.
      DataFileReader<double> input_file(filename, chunk_size);
      std::size_t n_cols = input_file.n_cols();
      _CovAccumulation acc(0, n_cols);
      Matrix<double> m = std::move(input_file.next_block());
      while (m.n_rows() > 0) {
        // periodic correction
        if (periodic) {
          FastPCA::deg2rad_inplace(m);
          FastPCA::Periodic::shift_matrix_columns_inplace(m
                                                        , shifts);
        }
        // center data
        FastPCA::shift_matrix_columns_inplace(m, means);
        // normalize columns by dividing by sigma
        if (use_correlation) {
          FastPCA::scale_matrix_columns_inplace(m
                                              , inverse_sigmas);
        }
        acc = _join_accumulations(acc, _accumulate_covariance(m));
        m = std::move(input_file.next_block());
      }
      // last, we calculate the total covariance from the
      // accumulated data.
      return _get_covariance(acc);
    }
  } // end local namespace


  namespace Periodic {
    // compute covariance matrix for periodic data
    SymmetricMatrix<double>
    covariance_matrix(const std::string filename
                    , const std::size_t max_chunk_size
                    , bool use_correlation
                    , Matrix<double> stats) {
      return _covariance_matrix(filename
                              , max_chunk_size
                              , use_correlation
                              , true
                              , stats);
    }
  } // end namespace FastPCA::Periodic

  SymmetricMatrix<double>
  covariance_matrix(const std::string filename
                  , const std::size_t max_chunk_size
                  , bool use_correlation
                  , Matrix<double> stats) {
    return _covariance_matrix(filename
                            , max_chunk_size
                            , use_correlation
                            , false
                            , stats);
  }
} // end namespace FastPCA


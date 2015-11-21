
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

    _CovAccumulation::_CovAccumulation(SymmetricMatrix<double> m, std::vector<double> sum, std::size_t n)
      : m(m),
        sum_observations(sum),
        n_observations(n) {
    }

    _CovAccumulation::_CovAccumulation(std::size_t n_observations, std::size_t n_observables)
      : m(SymmetricMatrix<double>(n_observables)),
        sum_observations(std::vector<double>(n_observables)),
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
        cc = 0.0;
        #pragma omp parallel for default(none)\
                                 firstprivate(i)\
                                 private(t)\
                                 shared(c,m)\
                                 reduction(+:cc)
        for (t = 0; t < nr; ++t) {
          cc += m(t,i);
        }
        c.sum_observations[i] = cc;
      }
      c.n_observations = nr;
      return c;
    }

    _CovAccumulation
    _join_accumulations(const CovAccumulation& c1, const CovAccumulation& c2) {
      assert(c1.m.n_cols() == c2.m.n_cols());
      SymmetricMatrix<double> joint_m = c1.m + c2.m;
      std::vector<double> joint_sum(c1.sum_observations.size());
      std::size_t joint_n_obs = c1.n_observations + c2.n_observations;
      std::transform(c1.sum_observations.begin(),
                     c1.sum_observations.end(),
                     c2.sum_observations.begin(),
                     joint_sum.begin(),
                     std::plus<double>());
      return {joint_m, joint_sum, joint_n_obs};
    }

    SymmetricMatrix<double>
    _get_covariance(const CovAccumulation& acc) {
      SymmetricMatrix<double> sm = acc.m;
      std::size_t i, j;
      for (i = 0; i < sm.n_cols(); ++i) {
        for (j = 0; j <= i; ++j) {
          sm(i,j) = (sm(i,j) - (acc.sum_observations[i] *
                                acc.sum_observations[j] /
                                (acc.n_observations)      )) / (acc.n_observations-1);
        }
      }
      return sm;
    }
    
    SymmetricMatrix<double>
    _covariance_matrix(const std::string filename
                     , const std::size_t max_chunk_size) {
      //TODO periodicity and normalization
      
      // take only half size because of intermediate results.
      std::size_t chunk_size = max_chunk_size / 2;
      // data files are often to big to read into RAM completely.
      // thus, we read the data blockwise, calculate the temporary
      // covariance data and accumulate the results.
      DataFileReader<double> input_file(filename, chunk_size);
      std::size_t n_available_cols = input_file.n_cols();
      CovAccumulation acc(0, n_available_cols);
      Matrix<double> m = std::move(input_file.next_block());
      while (m.n_rows() > 0) {
        acc = join_accumulations(acc, accumulate_covariance(m));
        m = std::move(input_file.next_block());
      }
      // last, we calculate the total covariance from the
      // accumulated data.
      return get_covariance(acc);
    }
  } // end local namespace


  namespace Periodic {
    // compute covariance matrix for periodic data
    SymmetricMatrix<double>
    covariance_matrix(const std::string filename
                    , const std::size_t max_chunk_size) {
      //TODO periodicity
      return _covariance_matrix(filename, max_chunk_size);
//      // because of the periodicity of the underlying data,
//      // we cannot use the summation trick we use above
//      // to accumulate the covariance in one pass.
//      // instead, we need to account for periodic boundary
//      // checks for every observation. therefore, we have to
//      // compute the means beforehand and compute the covariance
//      // in a second pass through the data.
//      std::size_t n_rows;
//      std::size_t n_cols;
//      std::vector<double> means;
//      std::tie(n_rows, n_cols, means) = FastPCA::Periodic::means(filename, max_chunk_size);
//      // compute covariance matrix using the precomputed means.
//      // for every expression (x_n - mean_n), check periodic boundaries
//      // before computing the product (x_i - mean_i)(x_j - mean_j).
//      SymmetricMatrix<double> cov(n_cols);
//      {
//        DataFileReader<double> input_file(filename, max_chunk_size);
//        Matrix<double> m = std::move(input_file.next_block());
//        std::size_t nr = m.n_rows();
//        std::size_t i, j, n;
//        while (nr > 0) {
//          //TODO shift by means
//          #pragma omp parallel for default(none)\
//                                   private(i,j,n)\
//                                   firstprivate(n_cols,nr,means,n_rows)\
//                                   shared(m,cov)
//          for (j=0; j < n_cols; ++j) {
//            for (i=0; i <= j; ++i) {
//              for (n=0; n < nr; ++n) {
//                cov(j,i) += FastPCA::angular_distance(m(n,j), means[j])
//                          * FastPCA::angular_distance(m(n,i), means[i])
//                          / (n_rows-1);
//              }
//            }
//          }
//          m = std::move(input_file.next_block());
//          nr = m.n_rows();
//        }
//      }
//      return cov;
    }
  } // end namespace FastPCA::Periodic

  SymmetricMatrix<double>
  covariance_matrix(const std::string filename
                  , const std::size_t max_chunk_size) {
    return _covariance_matrix(filename, max_chunk_size);
  }
} // end namespace FastPCA


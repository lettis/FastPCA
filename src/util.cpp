
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


#include "util.hpp"
#include "file_io.hpp"

#include <string>

namespace FastPCA {

  namespace {
    std::vector<double>
    _sigmas(const std::string filename
         , const std::size_t max_chunk_size
         , std::vector<double> means
         , bool periodic) {
      DataFileReader<double> input_file(filename, max_chunk_size);
      std::size_t n_cols = means.size();
      std::size_t n_rows = 0;
      std::vector<double> sigmas(n_cols, 0.0);
      Matrix<double> m = std::move(input_file.next_block());
      while (m.n_rows() > 0) {
        n_rows += m.n_rows();
        // subtract means
        if (periodic) {
          FastPCA::deg2rad_inplace(m);
          FastPCA::Periodic::shift_matrix_columns_inplace(m, means);
        } else {
          FastPCA::shift_matrix_columns_inplace(m, means);
        }
        // compute variances
        for (std::size_t i=0; i < m.n_rows(); ++i) {
          for (std::size_t j=0; j < n_cols; ++j) {
            sigmas[j] += m(i,j)*m(i,j);
          }
        }
        m = std::move(input_file.next_block());
      }
      // compute sigmas from variances
      for (std::size_t j=0; j < n_cols; ++j) {
        sigmas[j] = sqrt(sigmas[j] / (n_rows-1));
      }
      return sigmas;
    }
  } // end local namespace


  bool is_comment_or_empty(std::string line) {
    std::size_t pos = line.find_first_not_of(" ");
    if (pos == std::string::npos) {
      // empty string
      return true;
    } else if (line[pos] == '#' or line[pos] == '@') {
      // comment
      return true;
    } else {
      return false;
    }
  }

  std::tuple<std::size_t, std::size_t, std::vector<double>>
  means(const std::string filename
      , const std::size_t max_chunk_size) {
    DataFileReader<double> input_file(filename, max_chunk_size);
    std::size_t n_cols = input_file.n_cols();
    std::size_t n_rows = 0;
    std::vector<double> means(n_cols);
    Matrix<double> m = std::move(input_file.next_block());
    while (m.n_rows() > 0) {
      n_rows += m.n_rows();
      for (std::size_t i=0; i < m.n_rows(); ++i) {
        for (std::size_t j=0; j < n_cols; ++j) {
          means[j] += m(i,j);
        }
      }
      m = std::move(input_file.next_block());
    }
    for (std::size_t j=0; j < n_cols; ++j) {
      means[j] /= n_rows;
    }
    return std::make_tuple(n_rows, n_cols, means);
  }

  std::vector<double>
  sigmas(const std::string filename
       , const std::size_t max_chunk_size
       , std::vector<double> means) {
    return _sigmas(filename, max_chunk_size, means, false);
  }

  namespace Periodic {
    double
    distance(double theta1, double theta2) {
      double abs_diff = std::abs(theta1 - theta2);
      if (abs_diff <= M_PI) {
        return abs_diff;
      } else {
        return abs_diff - (2*M_PI);
      }
    }

    // compute circular means by averaging sines and cosines
    // and resolving the mean angle with the atan2 function.
    // additionally, return number of observations.
    std::tuple<std::size_t, std::size_t, std::vector<double>>
    means(const std::string filename
        , const std::size_t max_chunk_size) {
      DataFileReader<double> input_file(filename, max_chunk_size);
      Matrix<double> m = std::move(input_file.next_block());
      FastPCA::deg2rad_inplace(m);
      std::size_t i, j;
      std::size_t nr = m.n_rows();
      std::size_t nc = m.n_cols();
      std::vector<double> means(nc, 0.0);
      std::vector<double> means_sin(nc, 0.0);
      std::vector<double> means_cos(nc, 0.0);
      std::size_t n_rows_total = 0;
      while (nr > 0) {
        for (j=0; j < nc; ++j) {
          for (i=0; i < nr; ++i) {
            means_sin[j] += sin(m(i,j));
            means_cos[j] += cos(m(i,j));
          }
        }
        n_rows_total += nr;
        m = std::move(input_file.next_block());
        FastPCA::deg2rad_inplace(m);
        nr = m.n_rows();
      }
      for (j=0; j < nc; ++j) {
        means[j] = std::atan2(means_sin[j]/n_rows_total, means_cos[j]/n_rows_total);
      }
      return std::make_tuple(n_rows_total, nc, means);
    }

    std::vector<double>
    sigmas(const std::string filename
         , const std::size_t max_chunk_size
         , std::vector<double> means) {
      return _sigmas(filename, max_chunk_size, means, true);
    }
  } // end namespace FastPCA::Periodic
} // end namespace FastPCA


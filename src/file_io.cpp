
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


#include <cstdlib>
#include <set>

#include "file_io.hpp"
#include "covariance.hpp"

namespace FastPCA {

  AsciiLineWriter::AsciiLineWriter(std::string filename)
    : _filename(filename)
    , _fh(filename) {
  }

  void
  AsciiLineWriter::write(std::vector<double> v) {
    for (double val: v) {
      _fh << " " << val;
    }
    _fh << "\n";
  }

  FileType
  filename_suffix(const std::string filename) {
    const int equal = 0;
    const std::size_t pos_dot = filename.rfind(".");
    if (pos_dot == filename.npos) {
      return UNKNOWN;
    } else {
      const std::string suffix = filename.substr(pos_dot+1);
      if (suffix.compare("xtc") == equal) {
        return XTC;
      } else if (suffix.compare("gro") == equal) {
        return GRO;
      } else if (suffix.compare("pdb") == equal) {
        return PDB;
      } else {
        return UNKNOWN;
      }
    }
  }

  void
  calculate_projections(const std::string file_in,
                        const std::string file_out,
                        Matrix<double> eigenvecs,
                        std::size_t mem_buf_size,
                        bool use_correlation,
                        Matrix<double> stats) {
    // calculating the projection, we need twice the space
    // (original data + result)
    mem_buf_size /= 4;
    std::size_t n_variables = stats.n_rows();
    std::vector<double> means(n_variables);
    std::vector<double> inverse_sigmas(n_variables);
    for (std::size_t i=0; i < n_variables; ++i) {
      means[i] = stats(i,0);
      inverse_sigmas[i] = 1.0 / stats(i,1);
    }
    bool append_to_file = false;
    DataFileReader<double> fh_file_in(file_in, mem_buf_size);
    DataFileWriter<double> fh_file_out(file_out);
    read_blockwise(fh_file_in, [&](Matrix<double>& m) {
      FastPCA::shift_matrix_columns_inplace(m, means);
      if (use_correlation) {
        FastPCA::scale_matrix_columns_inplace(m, inverse_sigmas);
      }
      fh_file_out.write(std::move(m*eigenvecs), append_to_file);
      append_to_file = true;
    });
  }

  void
  read_blockwise(DataFileReader<double>& ifile
               , std::function<void(Matrix<double>&)> acc) {
    Matrix<double> m = std::move(ifile.next_block());
    while (m.n_rows() > 0) {
      acc(m);
      m = std::move(ifile.next_block());
    }
  }

  namespace Periodic {
    void
    calculate_projections(const std::string file_in,
                          const std::string file_out,
                          Matrix<double> eigenvecs,
                          std::size_t mem_buf_size,
                          bool use_correlation,
                          Matrix<double> stats) {
      mem_buf_size /= 4;
      std::size_t n_variables = stats.n_rows();
      std::vector<double> means(n_variables);
      std::vector<double> inverse_sigmas(n_variables);
      std::vector<double> shifts(n_variables);
      for (std::size_t i=0; i < n_variables; ++i) {
        means[i] = stats(i,0);
        inverse_sigmas[i] = 1.0 / stats(i,1);
        shifts[i] = stats(i,2);
      }
      // projections
      bool append_to_file = false;
      DataFileReader<double> fh_file_in(file_in, mem_buf_size);
      DataFileWriter<double> fh_file_out(file_out);
      read_blockwise(fh_file_in, [&](Matrix<double>& m) {
        // convert degrees to radians
        FastPCA::deg2rad_inplace(m);
        FastPCA::Periodic::shift_matrix_columns_inplace(m, shifts);
        if (use_correlation) {
          // scale data by sigmas for correlated projections
          FastPCA::scale_matrix_columns_inplace(m, inverse_sigmas);
        }
        // output
        fh_file_out.write(m*eigenvecs, append_to_file);
        append_to_file = true;
      });
    }
  } // end namespace FastPCA::Periodic
} // end namespace FastPCA


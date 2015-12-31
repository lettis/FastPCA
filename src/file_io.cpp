
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
                        bool use_correlation) {
    // calculating the projection, we need twice the space
    // (original data + result)
    mem_buf_size /= 4;
    bool append_to_file = false;
    std::vector<double> means;
    std::vector<double> sigmas;
    std::tie(std::ignore, std::ignore, means) = FastPCA::means(file_in, mem_buf_size);
    if (use_correlation) {
      sigmas = FastPCA::sigmas(file_in, mem_buf_size, means);
      for (double& s: sigmas) {
        s = 1.0 / s;
      }
    }
    DataFileReader<double> fh_file_in(file_in, mem_buf_size);
    DataFileWriter<double> fh_file_out(file_out);
    while ( ! fh_file_in.eof()) {
      Matrix<double> m = std::move(fh_file_in.next_block());
      if (m.n_rows() > 0) {
        FastPCA::shift_matrix_columns_inplace(m, means);
        if (use_correlation) {
          FastPCA::scale_matrix_columns_inplace(m, sigmas);
        }
        fh_file_out.write(std::move(m*eigenvecs), append_to_file);
        append_to_file = true;
      }
    }
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
                          bool use_dih_shifts) {
      mem_buf_size /= 4;
      std::vector<double> means;
      std::vector<double> sigmas;
      std::vector<double> dih_shifts;
      std::vector<double> scaled_periodicities;
      // compute means
      if ((!use_dih_shifts) || use_correlation) {
        std::tie(std::ignore, std::ignore, means) = FastPCA::Periodic::means(file_in, mem_buf_size);
      }
      // compute sigmas
      if (use_correlation) {
        sigmas = FastPCA::Periodic::sigmas(file_in, mem_buf_size, means);
        scaled_periodicities.resize(sigmas.size(), 2*M_PI);
        for (std::size_t j=0; j < sigmas.size(); ++j) {
          // invert sigmas for easier rescaling
          sigmas[j] = 1.0 / sigmas[j];
          scaled_periodicities[j] *= sigmas[j];
        }
      }
      // compute dih-shifts
      if (use_dih_shifts) {
        dih_shifts = FastPCA::Periodic::dih_shifts(file_in, mem_buf_size);
        if (use_correlation) {
          for (std::size_t i=0; i < dih_shifts.size(); ++i) {
            //TODO: check signs
            dih_shifts[i] = (means[i] - dih_shifts[i]) * sigmas[i];
          }
        }
      }
      // projections
      bool append_to_file = false;
      DataFileReader<double> fh_file_in(file_in, mem_buf_size);
      DataFileWriter<double> fh_file_out(file_out);
      read_blockwise(fh_file_in, [&](Matrix<double>& m) {
        // convert degrees to radians
        FastPCA::deg2rad_inplace(m);
        if (use_correlation || (! use_dih_shifts)) {
          // shift by periodic means
          FastPCA::Periodic::shift_matrix_columns_inplace(m, means);
        } else if (use_dih_shifts) {
          // shift by optimal dih-shifts
          FastPCA::Periodic::shift_matrix_columns_inplace(m, dih_shifts);
        }
        if (use_correlation) {
          // scale data by sigmas for correlated projections
          FastPCA::scale_matrix_columns_inplace(m, sigmas);
          if (use_dih_shifts) {
          //TODO: check if this is working
            FastPCA::Periodic::shift_matrix_columns_inplace(m, dih_shifts, scaled_periodicities);
          }
        }
        // output
        fh_file_out.write(m*eigenvecs, append_to_file);
        append_to_file = true;
      });
    }
  } // end namespace FastPCA::Periodic
} // end namespace FastPCA


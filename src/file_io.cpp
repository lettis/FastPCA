
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

#define PI (3.14159265358979323846264338327950288)

#include "file_io.hpp"

namespace FastPCA {

  namespace Periodic {
    namespace {
      void
      shift_matrix(Matrix<double>& m
                 , std::vector<double> shifts) {
        std::size_t n;
        std::size_t j;
        std::size_t nr = m.n_rows();
        std::size_t nc = m.n_cols();
        #pragma omp parallel for default(none) private(j,n) firstprivate(nc,nr,shifts) shared(m)
        for (n=0; n < nr; ++n) {
          for (j=0; j < nc; ++j) {
            m(n, j) += shifts[j];
          }
        }
      }

      void
      calculate_projections_shifted(const std::string file_in,
                                    AsciiLineWriter& fh_file_out,
                                    Matrix<double> eigenvecs,
                                    std::size_t mem_buf_size,
                                    std::vector<double> shifts,
                                    std::set<std::size_t>& seen_before) {
        mem_buf_size /= 4;
        auto is_in_unit_range = [](double theta) -> bool {return ((-PI < theta) && (theta <= PI));};
        DataFileReader<double> fh_file_in(file_in, mem_buf_size);
        Matrix<double> m = std::move(fh_file_in.next_block());
        FastPCA::deg2rad(m);
        shift_matrix(m, shifts);
        Matrix<double> proj;
        while (m.n_rows() > 0) {
          // compute projection
          proj = m * eigenvecs;
          for (std::size_t n=0; n < proj.n_rows(); ++n) {
            if (seen_before.count(n) == 0) {
              bool in_unit_range = true;
              for (std::size_t j=0; j < proj.n_cols(); ++j) {
                if ( ! is_in_unit_range(proj(n, j))) {
                  in_unit_range = false;
                  break;
                }
              }
              if (in_unit_range) {
                std::vector<double> buf_out(proj.n_cols());
                for (std::size_t j=0; j < proj.n_cols(); ++j) {
                  buf_out[j] = proj(n, j);
                }
                fh_file_out.write(buf_out);
                seen_before.insert(n);
              }
            }
          }
          m = std::move(fh_file_in.next_block());
          FastPCA::deg2rad(m);
          shift_matrix(m, shifts);
        }
      }
    } // end local namespace

    void
    calculate_projections(const std::string file_in,
                          const std::string file_out,
                          Matrix<double> eigenvecs,
                          std::size_t mem_buf_size) {
      std::size_t nc = eigenvecs.n_cols();
      AsciiLineWriter fh_file_out(file_out);
      std::set<std::size_t> seen_before;
      for (std::size_t j=0; j < nc; ++j) {
        calculate_projections_shifted(file_in, fh_file_out, eigenvecs, mem_buf_size, {    0,    0}, seen_before);
        calculate_projections_shifted(file_in, fh_file_out, eigenvecs, mem_buf_size, {    0, 2*PI}, seen_before);
        calculate_projections_shifted(file_in, fh_file_out, eigenvecs, mem_buf_size, {    0,-2*PI}, seen_before);
        calculate_projections_shifted(file_in, fh_file_out, eigenvecs, mem_buf_size, { 2*PI,    0}, seen_before);
        calculate_projections_shifted(file_in, fh_file_out, eigenvecs, mem_buf_size, { 2*PI, 2*PI}, seen_before);
        calculate_projections_shifted(file_in, fh_file_out, eigenvecs, mem_buf_size, { 2*PI,-2*PI}, seen_before);
        calculate_projections_shifted(file_in, fh_file_out, eigenvecs, mem_buf_size, {-2*PI,    0}, seen_before);
        calculate_projections_shifted(file_in, fh_file_out, eigenvecs, mem_buf_size, {-2*PI, 2*PI}, seen_before);
        calculate_projections_shifted(file_in, fh_file_out, eigenvecs, mem_buf_size, {-2*PI,-2*PI}, seen_before);
      }
    }
  }; // end namespace FastPCA::Periodic

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
                        std::size_t mem_buf_size) {
    // calculating the projection, we need twice the space
    // (original data + result)
    mem_buf_size /= 4;
    bool first_write = true;
    DataFileReader<double> fh_file_in(file_in, mem_buf_size);
    DataFileWriter<double> fh_file_out(file_out);
    while ( ! fh_file_in.eof()) {
      Matrix<double> m = std::move(fh_file_in.next_block());
      if (m.n_rows() > 0) {
        if (first_write) {
          // write projected data directly to file
          fh_file_out.write(std::move(m*eigenvecs));
          first_write = false;
        } else {
          // append next project block to file
          fh_file_out.write(std::move(m*eigenvecs), true);
        }
      }
    }
  }
} // end namespace FastPCA


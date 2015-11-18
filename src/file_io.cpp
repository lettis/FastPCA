
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
#include "covariance.hpp"

namespace FastPCA {

  namespace {
    void
    _whiten_data(const std::string file_in,
                 const std::string file_out,
                 SymmetricMatrix<double> cov,
                 std::size_t mem_buf_size,
                 bool periodic) {
      std::size_t i,j,nr;
      std::size_t n_rows;
      std::size_t n_cols;
      std::vector<double> means;
      if (periodic) {
        std::tie(n_rows, n_cols, means) = FastPCA::Periodic::means(file_in, mem_buf_size);
      } else {
        std::tie(n_rows, n_cols, means) = FastPCA::means(file_in, mem_buf_size);
      }
      std::vector<double> sigma(n_cols);
      for (std::size_t j=0; j < n_cols; ++j) {
        sigma[j] = std::sqrt(cov(j,j));
      }
      DataFileReader<double> fh_file_in(file_in, mem_buf_size);
      AsciiLineWriter fh_file_out(file_out);
      bool first_write = true;
      while ( ! fh_file_in.eof()) {
        Matrix<double> m = std::move(fh_file_in.next_block());
        if (periodic) {
          FastPCA::deg2rad(m);
        }
        nr = m.n_rows();
        if (nr > 0) {
          // convert block: shifted by mean, divided by sigma
          #pragma omp parallel for default(none)\
                                   private(i,j)\
                                   firstprivate(nr,n_cols,periodic)\
                                   shared(m,means,sigma)\
                                   collapse(2)
          for (i=0; i < nr; ++i) {
            for (j=0; j < n_cols; ++j) {
              m(i,j) = m(i,j) - means[j];
              if (periodic) {
                // periodic boundary checks
                if (m(i,j) < -PI) {
                  m(i,j) = m(i,j) + 2*PI;
                } else if (m(i,j) > PI) {
                  m(i,j) = m(i,j) - 2*PI;
                }
              }
              m(i,j) = m(i,j) / sigma[j];
            }
          }
          if (periodic) {
            FastPCA::rad2deg(m);
          }
          if (first_write) {
            // write projected data directly to file
            fh_file_out.write(m);
            first_write = false;
          } else {
            // append next project block to file
            fh_file_out.write(m, true);
          }
        }
      }
    }
  } // end local namespace

  namespace Periodic {
//    namespace {
//      void
//      shift_matrix(Matrix<double>& m
//                 , std::vector<double> shifts) {
//        std::size_t n;
//        std::size_t j;
//        std::size_t nr = m.n_rows();
//        std::size_t nc = m.n_cols();
//        #pragma omp parallel for default(none) private(j,n) firstprivate(nc,nr,shifts) shared(m)
//        for (n=0; n < nr; ++n) {
//          for (j=0; j < nc; ++j) {
//            m(n, j) += shifts[j];
//          }
//        }
//      }
//    } // end local namespace

    void
    calculate_projections(const std::string file_in,
                          AsciiLineWriter& fh_file_out,
                          Matrix<double> eigenvecs,
                          std::size_t mem_buf_size) {
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
          std::vector<double> buf_out(proj.n_cols());
          for (std::size_t j=0; j < proj.n_cols(); ++j) {
            buf_out[j] = proj(n, j);
          }
          fh_file_out.write(buf_out);
        }
        m = std::move(fh_file_in.next_block());
        FastPCA::deg2rad(m);
      }
    }

    void
    whiten_data(const std::string file_in,
                const std::string file_out,
                SymmetricMatrix<double> cov,
                std::size_t mem_buf_size,
                ) {
      _whiten_data(file_in, file_out, cov, mem_buf_size, true);
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

  void
  whiten_data(const std::string file_in,
              const std::string file_out,
              SymmetricMatrix<double> cov,
              std::size_t mem_buf_size,
              ) {
    _whiten_data(file_in, file_out, cov, mem_buf_size, false);
  }

} // end namespace FastPCA


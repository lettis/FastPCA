
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

#include "file_io.hpp"

#include <limits>


namespace FastPCA {
  namespace { // begin local namespace
  namespace UseCSV {
    template <class T>
    Matrix<T> read_chunk(LTS::AsciiParser<T>& parser, std::size_t n_lines) {
      std::size_t n_read_lines = n_lines;
      std::vector<T> m = std::move(parser.next_n_lines_continuous(n_read_lines,
                                                                  LTS::AsciiParser<T>::COL_MAJOR));
      return Matrix<T>(m, n_read_lines, parser.n_cols());
    }
    
    template <class T>
    void write_vector(std::ofstream& fh, std::vector<T> v, bool in_line) {
      if (fh.good()) {
        //int n_digits = std::numeric_limits<T>::digits10;
        int n_digits = 6;
        fh.precision(n_digits);
        fh.setf(std::ios::scientific);
        for (auto it=v.begin(); it != v.end(); ++it) {
          fh.width(n_digits+10); // 8 extra chars for spacing, +/- and scientific formatting
          if (in_line) {
            fh << " ";
          }
          fh << (double) *it;
          if ( ! in_line) {
            fh << std::endl;
          }
        }
        if (in_line) {
          fh << std::endl;
        }
      } else {
        throw FileNotFoundError();
      }
    }
  
    template <class T>
    void write_vector(std::string file_out, std::vector<T> v, bool append) {
      std::ofstream fh;
      if (append) {
        fh.open(file_out, std::ios::out | std::ios::app);
      } else {
        fh.open(file_out, std::ios::out);
      }
      write_vector(fh, v);
    }
  
    template <class T>
    void write_matrix(std::string file_out, Matrix<T> m, bool append) {
      std::ofstream fh;
      if (append) {
        fh.open(file_out, std::ios::out | std::ios::app);
      } else {
        fh.open(file_out, std::ios::out);
      }
      if (fh.is_open()) {
        //int n_digits = std::numeric_limits<T>::digits10;
        int n_digits = 6;
        fh.precision(n_digits);
        fh.setf(std::ios::scientific);
        for (std::size_t i=0; i < m.n_rows(); ++i) {
          for (std::size_t j=0; j < m.n_cols(); ++j) {
            fh.width(n_digits+8); // 8 extra chars for spacing, +/- and scientific formatting
            fh << m(i, j);
          }
          fh << std::endl;
        }
      } else {
        throw FileNotFoundError();
      }
    }
  } // end namespace UseCSV
  
  namespace UseXTC {
    RvecArray::RvecArray(std::size_t n_atoms)
      : n_atoms(n_atoms) {
      if (this->n_atoms > 0) {
        this->values = static_cast<rvec*>(calloc(this->n_atoms, sizeof(this->values[0])));
      }
    }
  
    RvecArray::~RvecArray() {
      free(this->values);
    }
  
    std::size_t n_atoms(std::string file_in) {
      int n_atoms=0;
      read_xtc_natoms(file_in.c_str(), &n_atoms);
      return (std::size_t) n_atoms;
    }
  
    template <class T>
    Matrix<T> read_chunk(std::shared_ptr<XDRFILE> file, std::size_t n_cols, std::size_t n_lines) {
      int n_fake_atoms = n_cols/3;
      int return_code = 0;
      int n_step = 0;
      float time_step = 1.0;
      matrix box_vec;
      RvecArray coords(n_fake_atoms);
      float prec = 1000.0;
      Matrix<T> m(n_lines, n_cols);
  
      std::size_t n_lines_0 = n_lines;
      for (; n_lines > 0; --n_lines) {
        // read next frame
        return_code = read_xtc(file.get(), n_fake_atoms, &n_step,
                               &time_step, box_vec, coords.values, &prec);
        if (return_code == exdrOK) {
          // add next frame to input matrix
          std::vector<T> r(n_cols);
          for (int i = 0; i < n_fake_atoms; ++i) {
            for (int j = 0; j < 3; ++j) {
              m(n_lines_0-n_lines, i*3+j) = (T) coords.values[i][j];
            }
          }
        } else if (return_code == exdrENDOFFILE) {
          //EOF: if no line has been read, just return an empty matrix.
          //     (i.e. return 'm' as initialized)
          break;
        } else if (return_code == exdrFILENOTFOUND) {
          throw std::runtime_error("xtc file not found");
          break;
        } else {
          throw std::runtime_error("strange error while reading xtc file");
          break;
        }
      }
      if (n_lines != 0) {
        // resize m if needed
        m = std::move(m.limited_rows(n_lines_0-n_lines));
      }
      return m;
    }
  
    template <class T>
    void write_vector(std::string file_out, std::vector<T> v, bool append) {
      // write vector into column, keeping two additional columns empty.
      // this is needed since xtc files only can store cartesian coordinates,
      // i.e. packets of three numbers.
      // vectors will be written in first 'x' field with values in different
      // rows (aka time-frames) to get it column-oriented as first column,
      // if xtc-data is dumped (e.g. with gmxdump).
      Matrix<T> m(v.size(), 3);
      for (std::size_t i=0; i < v.size(); ++i) {
        m(i,0) = v[i];
        m(i,1) = 0.0;
        m(i,2) = 0.0;
      }
      write_matrix(file_out, m, append);
    }
    
    template <class T>
    void write_matrix(std::string file_out, Matrix<T> m, bool append) {
      std::shared_ptr<XDRFILE> file(xdrfile_open(file_out.c_str(), append ? "a" : "w"),
                                    // deleter function called when pointer goes out of scope
                                    [=](XDRFILE* f) {
                                      xdrfile_close(f);
                                    });
      // faked box matrix for xtc-file
      float fake_box_matrix[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
      // xtc-files are only readable/writable in 3D cartesian coordinates
      // for atoms. if the number of matrix columns is not divisable by three,
      // add zero-valued columns to fill.
      int n_fake_atoms = m.n_cols() / 3;
      if (m.n_cols() % 3 == 0) {
        ++n_fake_atoms;
      }
      std::size_t n_fake_cols = 3*n_fake_atoms;
      std::size_t n_cols = m.n_cols();
      RvecArray buf(3*n_fake_atoms);
      for (std::size_t r=0; r < m.n_rows(); ++r) {
        for (std::size_t c=0; c < n_fake_cols; ++c) {
          if (c < n_cols) {
            buf.values[c/3][c%3] = m(r,c);
          } else {
            buf.values[c/3][c%3] = 0.0;
          }
        }
        write_xtc(file.get(), n_fake_atoms, r, 1.0, fake_box_matrix, buf.values, 1000.0);
      }
    }
  } // end UseXTC namespace
  } // end local namespace
  
  
  template <class T>
  DataFileReader<T>::DataFileReader(std::string filename, std::size_t buf_size_bytes)
    : _filename(filename),
      _buf_size_bytes(buf_size_bytes),
      _n_cols(0),
      _eof(false) {
    this->_ftype = filename_suffix(filename);
  
    if (this->_ftype == XTC) {
      // use XTC file
      //this->_fh_xtc = UseXTC::XTCFile(filename, "r");
      this->_fh_xtc = std::shared_ptr<XDRFILE>(xdrfile_open(filename.c_str(), "r"),
                                               // deleter function called when pointer goes out of scope
                                               [](XDRFILE* f) {
                                                 xdrfile_close(f);
                                               });
      this->_n_cols = 3*UseXTC::n_atoms(filename);
    } else {
      this->_parser.open(filename);
      this->_n_cols = this->_parser.n_cols();
    }
  }
  
  template <class T>
  DataFileReader<T>::DataFileReader(std::string filename)
    : DataFileReader(filename, FastPCA::gigabytes_to_bytes(1)) {
  }
  
  template <class T>
  Matrix<T> DataFileReader<T>::next_block(std::size_t n_rows) {
    std::size_t n_possible_rows;
    if (n_rows == 0){
      n_possible_rows = this->_buf_size_bytes / (sizeof(T) * this->_n_cols);
    } else {
      n_possible_rows = n_rows;
    }
    if (this->_ftype == XTC) {
      if (this->_n_cols == 0) {
        throw FileFormatError();
      }
      Matrix<T> m = UseXTC::read_chunk<T>(this->_fh_xtc, this->_n_cols, n_possible_rows);
      if (m.n_rows() == 0) {
        this->_eof = true;
        return Matrix<T>();
      } else {
        return m;
      }
    } else {
      // assume ASCII file
      if (this->_parser.eof()) {
        this->_eof = true;
        return Matrix<T>();
      } else {
        if (this->_n_cols == 0) {
          throw FileFormatError();
        }
        //return UseCSV::read_chunk<T>(this->_fh, this->_n_cols, n_possible_rows);
        return UseCSV::read_chunk<T>(this->_parser, n_possible_rows);
      }
    }
  }
  
  template <class T>
  std::size_t DataFileReader<T>::n_cols() {
    return this->_n_cols;
  }
  
  template <class T>
  bool DataFileReader<T>::eof() const {
    return this->_eof;
  }
  
  ////
  
  template <class T>
  DataFileWriter<T>::DataFileWriter(std::string filename)
    : _filename(filename) {
    this->_ftype = filename_suffix(filename);
  }
  
  template <class T>
  void DataFileWriter<T>::write(std::vector<T> v, bool append) {
    switch (this->_ftype) {
      case XTC:
        UseXTC::write_vector(this->_filename, v, append);
        break;
      case UNKNOWN:
        UseCSV::write_vector(this->_filename, v, append);
        break;
      default:
        throw std::runtime_error("unknown file type to write vector");
    }
  }
  
  template <class T>
  void DataFileWriter<T>::write(Matrix<T> m, bool append) {
    switch (this->_ftype) {
      case XTC:
        UseXTC::write_matrix(this->_filename, m, append);
        break;
      case UNKNOWN:
        UseCSV::write_matrix(this->_filename, m, append);
        break;
      default:
        throw std::runtime_error("unknown file type given to write matrix");
    }
  }

} // end namespace FastPCA


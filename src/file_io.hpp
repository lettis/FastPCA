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
#include <vector>
#include <string>
#include <fstream>
#include <memory>

#include "matrix.hpp"
#include "util.hpp"

#include "AsciiParser.hpp"

extern "C" {
  // use xdrfile library to read/write xtc and trr files from gromacs
  #include "xdrfile/xdrfile.h"
  #include "xdrfile/xdrfile_xtc.h"
  #include "xdrfile/xdrfile_trr.h"
}


namespace FastPCA {
  namespace {
    namespace UseCSV {
      template <class T>
      Matrix<T> read_chunk(std::ifstream& fh, std::size_t n_cols, std::size_t n_lines);

      template <class T>
      void write_vector(std::ofstream& fh, std::vector<T> v, bool in_line=false);

      template <class T>
      void write_vector(std::string file_out, std::vector<T> v, bool append);

      template <class T>
      void write_matrix(std::string file_out, Matrix<T> m, bool append);
    } // end namespace UseCSV

    namespace UseXTC {
      class RvecArray {
       public:
        RvecArray(std::size_t n_atoms);
        ~RvecArray();

        std::size_t n_atoms;
        rvec* values;
      };

      std::size_t n_atoms(std::string file_in);

      template <class T>
      Matrix<T> read_chunk(std::shared_ptr<XDRFILE> file, std::size_t n_cols, std::size_t n_lines);

      template <class T>
      void write_vector(std::string file_out, std::vector<T> v, bool append);

      template <class T>
      void write_matrix(std::string file_out, Matrix<T> m, bool append);
    } // end namespace UseXTC
  } // end local namespace


  enum FileType {
    UNKNOWN, XTC, GRO, PDB
  };

  template <class T>
  class DataFileReader {
   public:
    DataFileReader(std::string filename);
    DataFileReader(std::string filename, std::size_t buf_size_bytes);

    Matrix<T> next_block(std::size_t n_rows=0);
    std::size_t n_cols();

    bool eof() const;

   private:
    std::string _filename;
    std::size_t _buf_size_bytes;
    FileType _ftype;
    std::size_t _n_cols;
    bool _eof;
    //std::ifstream _fh;
    LTS::AsciiParser<T> _parser;
    std::shared_ptr<XDRFILE> _fh_xtc;
  };

  template <class T>
  class DataFileWriter {
   public:
    DataFileWriter(std::string filename);

    void write(std::vector<T> v, bool append=false);
    void write(Matrix<T> m, bool append=false);

   private:
    std::string _filename;
    FileType _ftype;
  };

  class AsciiLineWriter {
   public:
    AsciiLineWriter(std::string filename);
    void write(std::vector<double> v);
   private:
    std::string _filename;
    std::ofstream _fh;
  };


  FileType filename_suffix(const std::string filename);

  void calculate_projections(const std::string file_in,
                             const std::string file_out,
                             Matrix<double> eigenvecs,
                             std::size_t mem_buf_size,
                             bool use_correlation);

  namespace Periodic {
    void calculate_projections(const std::string file_in,
                               const std::string file_out,
                               Matrix<double> eigenvecs,
                               std::size_t mem_buf_size,
                               bool use_correlation);
  } // end namespace FastPCA::Periodic

} // end namespace FastPCA

// load template implementations
#include "file_io.hxx"


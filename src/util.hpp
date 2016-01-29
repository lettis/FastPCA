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

// resolve circular dependency
namespace FastPCA {
  // used in file_io
  constexpr std::size_t
  gigabytes_to_bytes(std::size_t gb) {
    return gb * 1073741824;
  }
  // used in matrix
  template <class T>
  std::vector<std::size_t>
  sorted_index(const std::vector<T> &v, bool reverse_sorting=false);
}

#include "matrix.hpp"
#include "file_io.hpp"

namespace FastPCA {

  bool
  is_comment_or_empty(std::string line);
  
  template <class T>
  std::vector<T>
  parse_line(std::string line);
  
  template <class T>
  void
  deg2rad_inplace(Matrix<T>& m);

  template <class T>
  void
  deg2rad_inplace(std::vector<T>& v);

  template <class T>
  void
  rad2deg_inplace(Matrix<T>& m);

  template <class T>
  void
  shift_matrix_columns_inplace(Matrix<T>& m, std::vector<T> shifts);

  template <class T>
  void
  scale_matrix_columns_inplace(Matrix<T>& m, std::vector<T> factors);

  // statistical attributes in three columns:
  //  - means
  //  - sigmas
  Matrix<double> stats(const std::string filename
                     , const std::size_t max_chunk_size);

  namespace Periodic {
    /**
     * angular distance between two angles
     */
    double
    distance(double theta1, double theta2);

    /**
     * return corresponding number restricted to
     * periodic interval inside boundaries of
     * [-periodicity/2, +periodicity/2].
     */
    template <class T>
    T
    normalized(T var, T periodicity);

    template <class T>
    void
    shift_matrix_columns_inplace(Matrix<T>& m
                               , std::vector<T> shifts
                               , std::vector<T> periodicities);

    template <class T>
    void
    shift_matrix_columns_inplace(Matrix<T>& m, std::vector<T> shifts);
  
    // statistical attributes in three columns:
    //  - means
    //  - sigmas
    //  - optimal shifts
    Matrix<double>
    stats(const std::string filename
        , const std::size_t max_chunk_size);
  } // end namespace FastPCA::Periodic

} // end namespace FastPCA

// template implementations
#include "util.hxx"


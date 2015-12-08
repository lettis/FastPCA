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


#include "matrix.hpp"

#include <vector>
#include <string>

namespace FastPCA {

  constexpr std::size_t
  gigabytes_to_bytes(std::size_t gb) {
    return gb * 1073741824;
  }
  
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

  /**
   * simple stats for observables:
   *   1. number of observations in data set
   *   2. number of observables
   *   3. means of observables
   */
  std::tuple<std::size_t, std::size_t, std::vector<double>>
  means(const std::string filename
      , const std::size_t max_chunk_size);

  std::vector<double>
  sigmas(const std::string filename
       , const std::size_t max_chunk_size
       , std::vector<double> means);

  namespace Periodic {
    /**
     * angular distance between two angles
     */
    double
    distance(double theta1, double theta2);

    template <class T>
    void
    shift_matrix_columns_inplace(Matrix<T>& m, std::vector<T> shifts);
  
    /**
     * simple stats for periodic observables:
     *   1. number of observations in data set
     *   2. number of observables
     *   3. means of observables
     */
    std::tuple<std::size_t, std::size_t, std::vector<double>>
    means(const std::string filename
        , const std::size_t max_chunk_size);

    std::vector<double>
    sigmas(const std::string filename
         , const std::size_t max_chunk_size
         , std::vector<double> means);

    /**
     * compute optimal shifts for dihedrals to
     * move low-sampled barrier to periodic boundary
     */
    std::vector<double>
    dih_shifts(const std::string filename
             , std::size_t max_chunk_size);
  } // end namespace FastPCA::Periodic

} // end namespace FastPCA

// template implementations
#include "util.hxx"


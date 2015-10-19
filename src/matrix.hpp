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
#include <iostream>

namespace FastPCA {

template <class T> class SymmetricMatrix;

template <class T>
class Matrix {
 public:
  Matrix();
  Matrix(Matrix<T>&& m);
  Matrix(const Matrix<T>& m);
  Matrix(std::size_t n_rows, std::size_t n_cols);
  // data must be given in column-major order
  Matrix(std::vector<T> data, std::size_t n_rows, std::size_t n_cols);
  Matrix(const SymmetricMatrix<T>& sym);
  
  T& operator()(const std::size_t i, const std::size_t j);
  const T& operator()(const std::size_t i, const std::size_t j) const;

  Matrix<T>& operator=(const Matrix<T>& m);
  Matrix<T>& operator=(Matrix<T>&& m);

  std::size_t n_rows() const;
  std::size_t n_cols() const;

  Matrix<T> operator*(const Matrix<T>& other);
  Matrix<T> limited_rows(std::size_t n_rows) const;

 private:
  std::vector<T> _m;
  std::size_t _n_rows;
  std::size_t _n_cols;
};

template <class T>
class SymmetricMatrix {
 public:
  SymmetricMatrix();
  SymmetricMatrix(std::size_t n);
  SymmetricMatrix(const Matrix<T>& m);

  T& operator()(const std::size_t i, const std::size_t j);
  const T& operator()(const std::size_t i, const std::size_t j) const;
  std::size_t n_rows() const;
  std::size_t n_cols() const;

  std::vector<T> eigenvalues();
  Matrix<T> eigenvectors();

  template <class T2>
  friend SymmetricMatrix<T2> operator+(const SymmetricMatrix<T2>& s1, const SymmetricMatrix<T2>& s2);

  template <class T2>
  friend std::ostream& operator<< (std::ostream& out, SymmetricMatrix<T2>& s);

 private:
  std::vector<T> _m;
  std::size_t _n_rows;

  bool _eigensystem_is_solved;
  void _solve_eigensystem();

  std::vector<T> _eigenvalues;
  Matrix<T> _eigenvectors;
};


namespace {

template <class T>
struct SyevData {
  // compute eigenvectors and eigenvalues ('V')
  // or eigenvalues only ('N')
  char jobz;
  // store symmetric matrix in upper (='U') or lower (='L') triangle.
  // beware: FORTRAN uses column-major matrix indices,
  // thus FORTRAN-matrices are transposed!
  char uplo;
  // matrix order
  int n;
  // array for input data; later contains eigenvectors
  std::vector<T> a;
  // leading dimension
  int lda;
  // array for eigenvalues (in ascending order)
  std::vector<T> w;
  // size of work array
  int lwork;
  // array for temporary data
  std::vector<T> work;
  // == 0: success, < 0: |info|-th argument had illegal value, > 0 convergence failure
  int info;
};

// set lwork = -1 to query for optimal workspace size.
// then re-prepare with dat.lwork = dat.work[0]
template <class T>
void _prepareSyevData(SyevData<T>& dat, SymmetricMatrix<T>& m, int lwork);

template <class T>
Matrix<T> _extractEigenvectors(const SyevData<T>& dat);

template <class T>
std::vector<T> _extractEigenvalues(const SyevData<T>& dat);

} // end local namespace

} // end namespace FastPCA

// template implementations
#include "matrix.hxx"


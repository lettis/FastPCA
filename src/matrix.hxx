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
#include "errors.h"
#include "matrix.h"

#include <utility>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <cassert>

namespace FastCA {

namespace {

template <class T>
void _prepareSyevData(SyevData<T>& dat, SymmetricMatrix<T>& m, int lwork) {
  // compute eigenvectors and eigenvalues
  dat.jobz = 'V';
  // store symmetric matrix in upper triangle.
  // our data will be written to lower triangle, but FORTRAN uses
  // column-major matrix indices, thus FORTRAN-matrices are transposed!
  dat.uplo = 'U';
  // matrix order
  dat.n = m.n_rows();
  // leading dimension; here equal to matrix order
  dat.lda = dat.n;
  // array for eigenvalues (in ascending order)
  dat.w.resize(dat.n, (T) 0);
  // size of work array
  dat.lwork = lwork;
  if (lwork > 0) {
    // array for temporary data
    dat.work.resize(lwork);
  } else {
    dat.work.resize(1);
  }
  // == 0: success, < 0: |info|-th argument had illegal value, > 0 convergence failure
  dat.info = 0;
  // array for input data; later contains eigenvectors
  dat.a.resize(dat.n*dat.n, 0);
  for (std::size_t i=0; i < dat.n; ++i) {
    for (std::size_t j=0; j < dat.n; ++j) {
      if (j > i) {
        dat.a[i*dat.n+j] = 0;
      } else {
        dat.a[i*dat.n+j] = m(i, j);
      }
    }
  }
}

template <class T>
Matrix<T> _extractEigenvectors(const SyevData<T>& dat) {
  // the eigenvectors from lapack are row-oriented
  // and ordered from lowest to highest eigenvalue.
  // we want a matrix with column-oriented eigenvectors
  // ordered from highest to lowest eigenvalue.
  Matrix<T> v(dat.n, dat.n);
  for (std::size_t i=0; i < dat.n; ++i) {
    for (std::size_t j=0; j < dat.n; ++j) {
      v(i,j) = dat.a[(dat.n-j-1)*dat.n+i];
    }
  }
  return v;
}

template <class T>
std::vector<T> _extractEigenvalues(const SyevData<T>& dat) {
  std::vector<T> eigenvals = dat.w;
  std::reverse(eigenvals.begin(), eigenvals.end());
  return eigenvals;
}

} // end local namespace

// Matrix
template <class T>
Matrix<T>::Matrix()
  : _n_rows(0),
    _n_cols(0) {
}

template <class T>
Matrix<T>::Matrix(const SymmetricMatrix<T>& sym) {
  this->_n_rows = sym.n_rows();
  this->_n_cols = sym.n_rows();
  this->_m = std::vector<T>(this->_n_cols*this->_n_rows);
  for (std::size_t i=0; i < this->_n_rows; ++i) {
    for (std::size_t j=0; j <= i ; ++j) {
      (*this)(i,j) = sym(i,j);
      if (i != j) {
        (*this)(j,i) = sym(i,j);
      }
    }
  }
}


template <class T>
Matrix<T>::Matrix(const Matrix<T>& other) {
  this->_n_rows = other._n_rows;
  this->_n_cols = other._n_cols;
  this->_m = other._m;
}


template <class T>
Matrix<T>::Matrix(Matrix<T>&& m)
  : _n_rows(0),
    _n_cols(0) {
  *this = std::move(m);
}

template <class T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& m) {
  // clear existing data
  this->_m.clear();
  // copy rhs values
  this->_m = m._m;
  this->_n_rows = m._n_rows;
  this->_n_cols = m._n_cols;
  // clear rhs object
  m._m.clear();
  m._n_rows = 0;
  m._n_cols = 0;
  return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& m) {
  this->_m = m._m;
  this->_n_rows = m._n_rows;
  this->_n_cols = m._n_cols;
  return *this;
}

template <class T>
Matrix<T>::Matrix(std::size_t n_rows, std::size_t n_cols)
  : _n_rows(n_rows),
    _n_cols(n_cols){
  this->_m = std::vector<T>(n_cols*n_rows);
}

template <class T>
Matrix<T>::Matrix(std::vector<T> data, std::size_t n_rows, std::size_t n_cols)
  : _n_rows(n_rows),
    _n_cols(n_cols),
    _m(data) {
}

template <class T>
inline T& Matrix<T>::operator()(const std::size_t i, const std::size_t j) {
  assert(i < this->_n_rows);
  assert(j < this->_n_cols);
  // addressing in column-major order for fast column traversal
  return this->_m[j*this->_n_rows+i];
}

template <class T>
inline const T& Matrix<T>::operator()(const std::size_t i, const std::size_t j) const {
  assert(i < this->_n_rows);
  assert(j < this->_n_cols);
  // addressing in column-major order for fast column traversal
  return this->_m[j*this->_n_rows+i];
}


template <class T>
inline std::size_t Matrix<T>::n_rows() const {
  return this->_n_rows;
}

template <class T>
inline std::size_t Matrix<T>::n_cols() const {
  return this->_n_cols;
}

template <class T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) {
  if (other._n_rows != this->_n_cols) {
    throw MatrixNotAlignedError();
  } else {
    Matrix<T> c(this->_n_rows, other._n_cols);
    //TODO: test impact of this (with and without collapse)
    std::size_t i, j, k;
    #pragma omp parallel for collapse(2) default(shared) private(i,j,k)
    for (i=0; i < this->_n_rows; ++i) {
      for (j=0; j < this->_n_cols; ++j) {
        for (k=0; k < this->_n_cols; ++k) {
          c(i,j) += (*this)(i,k) * other(k,j);
        }
      }
    }
    return c;
  }
}

template <class T>
Matrix<T> Matrix<T>::limited_rows(std::size_t n_rows) const {
  Matrix<T> m(n_rows, this->_n_cols);
  for (std::size_t i=0; i < n_rows; ++i) {
    for (std::size_t j=0; j < this->_n_cols; ++j) {
      m(i,j) = (*this)(i,j);
    }
  }
  return m;
}

// SymmetricMatrix

template <class T>
SymmetricMatrix<T>::SymmetricMatrix()
  : _n_rows(0),
    _eigensystem_is_solved(false) {
}

template <class T>
SymmetricMatrix<T>::SymmetricMatrix(std::size_t n)
  : _n_rows(n),
    _eigensystem_is_solved(false) {
  // allocate memory for n * (n+1) / 2 elements of T (lower triangle of matrix)
  this->_m = std::vector<T>(n*(n+1)/2);
}

template <class T>
SymmetricMatrix<T>::SymmetricMatrix(const Matrix<T>& m)
  : _n_rows(m.n_cols()),
    _eigensystem_is_solved(false) {
  std::size_t n = this->_n_rows;
  this->_m = std::vector<T>(n*(n+1)/2);
  for (std::size_t i=0; i < this->_n_rows; ++i) {
    for (std::size_t j=0; j <= i; ++j) {
      (*this)(i,j) = m(i,j);
    }
  }
}

template <class T>
T& SymmetricMatrix<T>::operator()(std::size_t i, std::size_t j) {
// store data as lower triangular matrix; addressing begins in top-left corner.
//   *
//   * *
//   * * *
//   * * * *
//   [...]
  if (j > i) {
    std::swap(i, j);
  }
  assert(i < this->_n_rows);
  // this is needed, since returning a reference may lead to change in data
  this->_eigensystem_is_solved = false;
  return this->_m[i*(i+1)/2 + j];
}

template <class T>
const T& SymmetricMatrix<T>::operator()(std::size_t i, std::size_t j) const {
// store data as lower triangular matrix; addressing begins in top-left corner.
//   *
//   * *
//   * * *
//   * * * *
//   [...]
  if (j > i) {
    std::swap(i, j);
  }
  assert(i < this->_n_rows);
  // this is needed, since returning a reference may lead to change in data
  return this->_m[i*(i+1)/2 + j];
}

template <class T>
std::size_t SymmetricMatrix<T>::n_rows() const {
  return this->_n_rows;
}

template <class T>
std::size_t SymmetricMatrix<T>::n_cols() const {
  return this->_n_rows;
}

template <class T>
std::vector<T> SymmetricMatrix<T>::eigenvalues() {
  if ( ! this->_eigensystem_is_solved) {
    this->_solve_eigensystem();
  }
  return this->_eigenvalues;
}

template <class T>
Matrix<T> SymmetricMatrix<T>::eigenvectors() {
  if ( ! this->_eigensystem_is_solved) {
    this->_solve_eigensystem();
  }
  return this->_eigenvectors;
}

template <class T>
SymmetricMatrix<T> operator+(const SymmetricMatrix<T>& s1, const SymmetricMatrix<T>& s2) {
  assert(s1.n_rows() == s2.n_rows());
  std::size_t nr = s1.n_rows();
  SymmetricMatrix<T> r(nr);
  std::transform(s1._m.begin(), s1._m.end(), s2._m.begin(), r._m.begin(), std::plus<T>());
  return r;
}

template <class T>
std::ostream& operator<< (std::ostream& out, SymmetricMatrix<T>& s) {
  const int precision = 10;
  for (std::size_t r = 0; r < s.n_rows(); ++r) {
    for (std::size_t c = 0; c < s.n_cols(); ++c) {
      out << std::setw(precision+4) << std::setprecision(precision) << std::fixed << s(r,c);
    }
    if (r != s.n_rows()-1) {
      out << std::endl;
    }
  }
  return out;
}

} // end namespace FastCA


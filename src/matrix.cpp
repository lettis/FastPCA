
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


#include "matrix.h"

namespace FastCA {

namespace {

extern "C" {
  // LAPACK SSYEV and DSYEV prototypes
  void ssyev_(char* jobz, char* uplo, int* n,
              float* a, int* lda, float* w,
              float* work, int* lwork, int*info);
  void dsyev_(char* jobz, char* uplo, int* n,
              double* a, int* lda, double* w,
              double* work, int* lwork, int*info);
}

} // end local namespace

template <>
void SymmetricMatrix<float>::_solve_eigensystem() {
  SyevData<float> dat;
  // prepare with lwork=-1: probe for optimal workspace size
  _prepareSyevData<float>(dat, (*this), -1);
  ssyev_(&dat.jobz, &dat.uplo, &dat.n, &dat.a[0], &dat.lda, &dat.w[0], &dat.work[0], &dat.lwork, &dat.info);
  // run syev
  _prepareSyevData<float>(dat, (*this), dat.work[0]);
  ssyev_(&dat.jobz, &dat.uplo, &dat.n, &dat.a[0], &dat.lda, &dat.w[0], &dat.work[0], &dat.lwork, &dat.info);
  // extract data
  this->_eigenvalues = _extractEigenvalues<float>(dat);
  this->_eigenvectors = _extractEigenvectors<float>(dat);
}

template <>
void SymmetricMatrix<double>::_solve_eigensystem() {
  SyevData<double> dat;
  // prepare with lwork=-1: probe for optimal workspace size
  _prepareSyevData<double>(dat, (*this), -1);
  dsyev_(&dat.jobz, &dat.uplo, &dat.n, &dat.a[0], &dat.lda, &dat.w[0], &dat.work[0], &dat.lwork, &dat.info);
  // run syev
  _prepareSyevData<double>(dat, (*this), dat.work[0]);
  dsyev_(&dat.jobz, &dat.uplo, &dat.n, &dat.a[0], &dat.lda, &dat.w[0], &dat.work[0], &dat.lwork, &dat.info);
  // extract data
  this->_eigenvalues = _extractEigenvalues<double>(dat);
  this->_eigenvectors = _extractEigenvectors<double>(dat);
}

} // end namespace FastCA


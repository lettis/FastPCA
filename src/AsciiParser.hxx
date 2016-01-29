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
#include <limits>
#include <sstream>

namespace LTS {

template <typename T>
AsciiParser<T>::AsciiParser()
  : _n_cols(0) {
}

template <typename T>
AsciiParser<T>::AsciiParser(const std::string filename)
  : _n_cols(0) {
  this->_fh_in.open(filename.c_str(), std::ios::in);
  this->_n_cols_from_first_line();
  this->_fh_in.seekg(this->_fh_in.beg);
}

template <typename T>
AsciiParser<T>::~AsciiParser() {
  this->_fh_in.close();
}

template <typename T>
void AsciiParser<T>::open(const std::string filename) {
  this->_fh_in.open(filename.c_str(), std::ios::in);
  this->_n_cols_from_first_line();
  this->_fh_in.seekg(this->_fh_in.beg);
}

template <typename T>
std::size_t AsciiParser<T>::n_cols() {
  return this->_n_cols;
}

template <typename T>
T AsciiParser<T>::next() {
  T buf=0;
  while (this->_comments_ignored()) {
    this->_fh_in >> buf;
  }
  return buf;
}

template <typename T>
std::vector<T> AsciiParser<T>::next_line() {
  while (this->_comments_ignored()) {
    std::vector<T> res(this->_n_cols);
    for (std::size_t j=0; j < this->_n_cols; ++j) {
      this->_fh_in >> res[j];
    }
    return res;
  }
}

template <typename T>
std::vector< std::vector<T> > AsciiParser<T>::next_n_lines(std::size_t n) {
  std::vector< std::vector<T> > res(n, std::vector<T>(this->_n_cols));
  std::size_t i=0;
  while (this->_comments_ignored()) {
    if (i < n) {
      for (std::size_t j=0; j < this->_n_cols; ++j) {
        this->_fh_in >> res[i][j];
      }
      ++i;
    } else {
      break;
    }
  }
  return res;
}

template <typename T>
std::vector<T> AsciiParser<T>::next_n_lines_continuous(std::size_t& n, AsciiParser::Flags mode) {
  // use result-vector directly if n is known
  std::vector<T> res(n * _n_cols);
  // use this, if n == 0, i.e. if number of rows is unknown
  std::vector<std::vector<T>> res_infty_n(_n_cols, std::vector<T>());
  std::size_t i=0;
  while (this->_comments_ignored()) {
    if (n == 0) {
      // read complete file
      for (std::size_t j=0; j < _n_cols; ++j) {
        T buf;
        _fh_in >> buf;
        res_infty_n[j].push_back(buf);
      }
    } else if (i < n) {
      // read until max number of rows is reached
      for (std::size_t j=0; j < _n_cols; ++j) {
        if (mode == ROW_MAJOR) {
          this->_fh_in >> res[i*_n_cols+j];
        } else if (mode == COL_MAJOR) {
          this->_fh_in >> res[j*n+i];
        }
      }
    } else {
      break;
    }
    ++i;
  }
  // handle case that less than n lines have been read
  if (i != n) {
    if (mode == ROW_MAJOR) {
      // trivial, just shorten vector
      res.resize(i * _n_cols);
      if (n == 0) {
        for (std::size_t ii=0; ii < i; ++ii) {
          for (std::size_t jj=0; jj < _n_cols; ++jj) {
            res[ii*_n_cols+jj] = res_infty_n[jj][ii];
          }
        }
      }
    } else if (mode == COL_MAJOR) {
      // recopy to new, smaller vector
      std::vector<T> buf(i * _n_cols);
      for (std::size_t ii=0; ii < i; ++ii) {
        for (std::size_t jj=0; jj < _n_cols; ++jj) {
          if (n == 0) {
            buf[jj*i+ii] = res_infty_n[jj][ii];
          } else {
            buf[jj*i+ii] = res[jj*n+ii];
          }
        }
      }
      res = buf;
    }
    n = i;
  }
  return res;
}

template <typename T>
bool AsciiParser<T>::eof() {
  return this->_fh_in.eof();
}

//// private

template <typename T>
void AsciiParser<T>::_n_cols_from_first_line() {
  while(this->_fh_in) {
    char c = this->_fh_in.peek();
    if (c == '#' || c == '\n' || c == '@') {
      this->_fh_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    } else {
      std::string buf, fake;
      std::getline(this->_fh_in, buf);
      std::stringstream ss(buf, std::stringstream::in);
      this->_n_cols = 0;
      while (ss >> fake) {
        this->_n_cols++;
      }
      break;
    }
  }
}

template <typename T>
bool AsciiParser<T>::_comments_ignored() {
  while (this->_fh_in) {
    char c = this->_fh_in.peek();
    if (c == '#' || c == '\n' || c == '@') {
      this->_fh_in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    } else {
      break;
    }
  }
  return this->_fh_in.good();
}

} // end namespace LTS


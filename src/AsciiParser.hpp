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
#include <fstream>
#include <string>
#include <vector>

namespace LTS {

template <typename T>
class AsciiParser {
 public:
  enum Flags {
    ROW_MAJOR,
    COL_MAJOR
  };
  
  AsciiParser();
  AsciiParser(const std::string filename);
  ~AsciiParser();
  // open file for reading
  void open(const std::string filename);
  // return number of columns in file
  std::size_t n_cols();
  // return next value
  T next();
  // return next line
  std::vector<T> next_line();
  // return n lines as STL vector of lines
  // TODO n as reference
  std::vector<std::vector<T>> next_n_lines(std::size_t n);
  // return n lines coninues in one STL vector
  //  addressing in ROW_MAJOR format: elem(i,j) = block[i*n_cols+j]
  //  addressing in COL_MAJOR format: elem(i,j) = block[j*n+i]
  //  if less then n lines can be read (e.g. in case of EOF), n will
  //  be reset to the actual number of lines read.
  //  If n is initialized to 0, read all lines until EOF.
  std::vector<T> next_n_lines_continuous(std::size_t& n, Flags mode=AsciiParser::ROW_MAJOR);
  // EOF reached?
  bool eof();
 private:
  // parse first line that is not empty
  // nor comment to get number of columns
  void _n_cols_from_first_line();
  // cycle through lines in opened file, halting every
  // time there are actual values and not just empty lines
  // or comments.
  // used internally as helper function for linewise parsing.
  bool _comments_ignored();
  std::ifstream _fh_in;
  std::size_t _n_cols;
};

} // end namespace LTS

#include "AsciiParser.hxx"


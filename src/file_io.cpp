
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

#include "file_io.h"

namespace FastCA {

FileType filename_suffix(const std::string filename) {
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

void calculate_projections(const std::string file_in,
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
      // write projected data directly to file
      if (first_write) {
        fh_file_out.write(std::move(m*eigenvecs));
        first_write = false;
      } else {
        fh_file_out.write(std::move(m*eigenvecs), true);
      }
    }
  }
}

} // end namespace FastCA


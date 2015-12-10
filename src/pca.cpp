
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


#include "covariance.hpp"
#include "file_io.hpp"
#include "matrix.hpp"
#include "util.hpp"

#include <string>
#include <omp.h>
#include <boost/program_options.hpp>

namespace b_po = boost::program_options;

int main(int argc, char* argv[]) {
  bool verbose = false;
  bool periodic = false;
  bool dih_shift = false;
  bool use_correlation = false;

  b_po::options_description desc (std::string(argv[0]).append(
    "\n\n"
    "Calculate principle components from large data files.\n"
    "Input data should be given as textfiles\n"
    "with whitespace separated columns or, alternatively as GROMACS .xtc-files.\n"
    "\n"
    "options"));

  desc.add_options()
    ("help,h", "show this help")
    // input
    ("file,f", b_po::value<std::string>(),
        "input (required): either whitespace-separated ASCII or GROMACS xtc-file.")
    ("cov-in,C", b_po::value<std::string>()->default_value(""),
        "input (optional): file with already calculated covariance matrix")
    ("vec-in", b_po::value<std::string>()->default_value(""),
        "input (optional): file with already computed eigenvecctors")
    // output
    ("proj,p", b_po::value<std::string>()->default_value(""),
        "output (optional): file for projected data")
    ("vec,v", b_po::value<std::string>()->default_value(""),
        "output (optional): file for eigenvectors")
    ("val,V", b_po::value<std::string>()->default_value(""),
        "output (optional): file for eigenvalues")
    ("cov,c", b_po::value<std::string>()->default_value(""),
        "output (optional): file for covariance matrix")
    ("norm,N", b_po::value(&use_correlation)->zero_tokens(),
        "if set, use correlation instead of covariance by normalizing input (default: false)")
    // parameters
    ("buf,b", b_po::value<std::size_t>()->default_value(2),
        "max. allocatable RAM [Gb] (default: 2)")
    ("periodic,P", b_po::value(&periodic)->zero_tokens(),
        "compute covariance and PCA on a torus (i.e. for periodic data like dihedral angles)")
    ("dih-shift", b_po::value(&dih_shift)->zero_tokens(),
        "shift barrier region of dihedrals to periodic border to minimize projection errors")
    ("verbose", b_po::value(&verbose)->zero_tokens(),
        "verbose mode (default: not set)")
    ("nthreads,n", b_po::value<std::size_t>()->default_value(0),
        "number of OpenMP threads to use. if set to zero, will use value of OMP_NUM_THREADS (default: 0)");

  b_po::variables_map args;
  try {
    b_po::store(b_po::parse_command_line(argc, argv, desc), args);
  } catch (b_po::error e) {
    std::cerr << e.what() << std::endl;
  }
  b_po::notify(args);

  if (args.count("help")) {
    std::cerr << desc << std::endl;
    return 1;
  }

  ////

  if (args.count("file")) {
    bool projection_file_given = (args["proj"].as<std::string>().compare("") != 0);
    bool eigenvec_file_given = (args["vec"].as<std::string>().compare("") != 0);
    bool eigenval_file_given = (args["val"].as<std::string>().compare("") != 0);
    bool covmat_file_given = (args["cov"].as<std::string>().compare("") != 0);
    bool input_covmat_file_given = (args["cov-in"].as<std::string>().compare("") != 0);
    bool input_eigenvec_file_given = (args["vec-in"].as<std::string>().compare("") != 0);

    if ( projection_file_given
      || eigenval_file_given
      || eigenvec_file_given
      || covmat_file_given) {

      std::size_t nthreads = args["nthreads"].as<std::size_t>();
      if (nthreads > 0) {
        // if not set explicitly (i.e. nthreads == 0), num_threads will be initialized to value of
        // environment variable OMP_NUM_THREADS
        omp_set_num_threads(nthreads);
      }
      std::string file_input = args["file"].as<std::string>();
      std::size_t mem_buf_size = FastPCA::gigabytes_to_bytes(args["buf"].as<std::size_t>());
      FastPCA::SymmetricMatrix<double> s;
      FastPCA::Matrix<double> vecs;
      if (input_covmat_file_given) {
        verbose && std::cerr << "loading covariance matrix from file" << std::endl;
        FastPCA::DataFileReader<double> cov_in(args["cov-in"].as<std::string>());
        s = FastPCA::SymmetricMatrix<double>(cov_in.next_block(cov_in.n_cols()));
      } else {
        if (input_eigenvec_file_given) {
          verbose && std::cerr << "loading eigenvectors from file" << std::endl;
          std::ifstream ifs(args["vec-in"].as<std::string>());
          int i=0;
          int n_cols = -1;
          while(ifs.good()) {
            std::string buf;
            std::getline(ifs, buf);
            if (! ifs.eof()) {
              std::vector<double> v = FastPCA::parse_line<double>(buf);
              if (n_cols < 0) {
                n_cols = v.size();
                vecs = FastPCA::Matrix<double>(n_cols, n_cols);
              }
              for (int j=0; j < n_cols; ++j) {
                vecs(i,j) = v[j];
              }
              ++i;
            }
          }
        } else {
          if (periodic) {
            verbose && ( ! use_correlation) && std::cerr << "constructing covariance matrix for periodic data" << std::endl;
            verbose &&     use_correlation  && std::cerr << "constructing correlation matrix for periodic data" << std::endl;
            s = FastPCA::Periodic::covariance_matrix(file_input, mem_buf_size, use_correlation);
          } else {
            verbose && ( ! use_correlation) && std::cerr << "constructing covariance matrix" << std::endl;
            verbose &&     use_correlation  && std::cerr << "constructing correlation matrix" << std::endl;
            s = FastPCA::covariance_matrix(file_input, mem_buf_size, use_correlation);
          }
          if (covmat_file_given) {
            verbose && ( ! use_correlation) && std::cerr << "writing covariance matrix" << std::endl;
            verbose &&     use_correlation  && std::cerr << "writing correlation matrix" << std::endl;
            std::string covmat_file = args["cov"].as<std::string>();
            FastPCA::DataFileWriter<double>(covmat_file).write(FastPCA::Matrix<double>(s));
          }
          if (eigenval_file_given) {
            verbose && std::cerr << "solving eigensystem/writing eigenvalues matrix" << std::endl;
            std::string eigenval_file = args["val"].as<std::string>();
            FastPCA::DataFileWriter<double>(eigenval_file).write(s.eigenvalues());
          }
          if (eigenvec_file_given) {
            verbose && std::cerr << "solving eigensystem/writing eigenvectors matrix" << std::endl;
            std::string eigenvec_file = args["vec"].as<std::string>();
            FastPCA::DataFileWriter<double>(eigenvec_file).write(s.eigenvectors());
          }
        }
      }
      if (projection_file_given) {
        std::string projection_file = args["proj"].as<std::string>();
        if (! input_eigenvec_file_given) {
          vecs = s.eigenvectors();
        }
        if (periodic) {
          std::vector<double> dih_shifts;
          verbose && std::cerr << "computing projections for periodic data" << std::endl;
          FastPCA::Periodic::calculate_projections(file_input, projection_file, vecs, mem_buf_size, use_correlation, dih_shift);
        } else {
          verbose && std::cerr << "computing projections" << std::endl;
          FastPCA::calculate_projections(file_input, projection_file, vecs, mem_buf_size, use_correlation);
        }
      }
    } else {
      std::cerr << "please specify at least one output file!" << std::endl;
      std::cerr << desc << std::endl;
      return EXIT_FAILURE;
    }
  } else {
    std::cerr << desc << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}



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
        "input (required): either whitespace-separated ASCII"
        " or GROMACS xtc-file.")
    ("cov-in,C", b_po::value<std::string>()->default_value(""),
        "input (optional): file with already calculated covariance matrix")
    ("vec-in,V", b_po::value<std::string>()->default_value(""),
        "input (optional): file with already computed eigenvectors")
    ("stats-in,S", b_po::value<std::string>()->default_value(""),
        "input (optional): mean values, sigmas and boundary shifts (shifts only for periodic)."
        " Provide this, if you want to project new data onto a previously computed principal space."
        " If you do not define the stats of the previous run, means, sigmas and shifts"
        " will be re-computed and the resulting projections will not be comparable"
        " to the previous ones.")
    // output
    ("proj,p", b_po::value<std::string>()->default_value(""),
        "output (optional): file for projected data")
    ("cov,c", b_po::value<std::string>()->default_value(""),
        "output (optional): file for covariance matrix")
    ("vec,v", b_po::value<std::string>()->default_value(""),
        "output (optional): file for eigenvectors")
    ("stats,s", b_po::value<std::string>()->default_value(""),
        "output (optional): mean values, sigmas and boundary shifts (shifts only for periodic)")
    ("val,l", b_po::value<std::string>()->default_value(""),
        "output (optional): file for eigenvalues")
    // parameters
    ("norm,N", b_po::value(&use_correlation)->zero_tokens(),
        "if set, use correlation instead of covariance by normalizing input (default: false)")
    ("buf,b", b_po::value<std::size_t>()->default_value(2),
        "max. allocatable RAM [Gb] (default: 2); WARNING there is some error in the code "
        "that leads to twice mem consumption at some point. use only HALF of what you would "
        "normally use. will be fixed soon")
    ("periodic,P", b_po::value(&periodic)->zero_tokens(),
        "compute covariance and PCA on a torus (i.e. for periodic data like dihedral angles)")
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

  try {
    if (args.count("file")) {
      auto check_file_given = [&args](std::string opt) {
        return (args[opt].as<std::string>().compare("") != 0);
      };
      bool projection_file_given = check_file_given("proj");
      bool eigenvec_file_given = check_file_given("vec");
      bool eigenval_file_given = check_file_given("val");
      bool covmat_file_given = check_file_given("cov");
      bool stats_file_given = check_file_given("stats");
      bool input_covmat_file_given = check_file_given("cov-in");
      bool input_eigenvec_file_given = check_file_given("vec-in");
      bool input_stats_file_given = check_file_given("stats-in");
  
      if ( projection_file_given
        || eigenval_file_given
        || eigenvec_file_given
        || covmat_file_given
        || stats_file_given) {
  
        std::size_t nthreads = args["nthreads"].as<std::size_t>();
        if (nthreads > 0) {
          // if not set explicitly (i.e. nthreads == 0), num_threads will
          // be initialized to value of environment variable OMP_NUM_THREADS
          omp_set_num_threads(nthreads);
        }
        std::string file_input = args["file"].as<std::string>();
        std::size_t mem_buf_size = FastPCA::gigabytes_to_bytes(
                                      args["buf"].as<std::size_t>());
        FastPCA::SymmetricMatrix<double> cov;
        FastPCA::Matrix<double> vecs;
        FastPCA::Matrix<double> stats;
        //// input
        if (input_stats_file_given) {
          // -S ?
          verbose && std::cerr << "loading stats (shifts, means, sigmas) from file" << std::endl;
          FastPCA::DataFileReader<double> fh_shifts(args["stats-in"].as<std::string>());
          stats = fh_shifts.next_block();
        } else {
          verbose && std::cerr << "computing stats (means, sigmas, ...)" << std::endl;
          if (periodic) {
            stats = FastPCA::Periodic::stats(file_input, mem_buf_size);
          } else {
            stats = FastPCA::stats(file_input, mem_buf_size);
          }
        }
        if (input_eigenvec_file_given && input_covmat_file_given) {
          // -V and -C ?
          std::cerr << "error: Providing the covariance matrix and the eigenvectors does not make sense." << std::endl
                    << "       If we have the vectors, we do not need the covariance matrix," << std::endl
                    << "       therefore the options '-C' and '-V' are mutually exclusive." << std::endl;
          return EXIT_FAILURE;
        } else if (input_eigenvec_file_given) {
          // -V ?
          verbose && std::cerr << "loading eigenvectors from file" << std::endl;
          FastPCA::DataFileReader<double> fh_vecs(args["vec-in"].as<std::string>());
          vecs = fh_vecs.next_block();
          if (projection_file_given && (! input_stats_file_given)) {
            std::cerr << "warning: You specified a projection output and gave" << std::endl
                      << "         pre-computed eigenvectors." << std::endl
                      << "         Projection is, however, centered to the mean." << std::endl
                      << "         Without stats (means, sigma, [dihedral shifts])," << std::endl
                      << "         you will in general get non-comparable results" << std::endl
                      << "         to previous projections!" << std::endl;
          }
        } else {
          if (input_covmat_file_given) {
            // -C ?
            verbose && std::cerr << "loading covariance matrix from file" << std::endl;
            FastPCA::DataFileReader<double> cov_in(args["cov-in"].as<std::string>());
            cov = FastPCA::SymmetricMatrix<double>(cov_in.next_block(cov_in.n_cols()));
          } else {
            if (periodic) {
              verbose && ( ! use_correlation) && std::cerr << "constructing covariance matrix for periodic data" << std::endl;
              verbose &&     use_correlation  && std::cerr << "constructing correlation matrix for periodic data" << std::endl;
              cov = FastPCA::Periodic::covariance_matrix(file_input, mem_buf_size, use_correlation, stats);
            } else {
              verbose && ( ! use_correlation) && std::cerr << "constructing covariance matrix" << std::endl;
              verbose &&     use_correlation  && std::cerr << "constructing correlation matrix" << std::endl;
              cov = FastPCA::covariance_matrix(file_input, mem_buf_size, use_correlation, stats);
            }
          }
          vecs = cov.eigenvectors();
        }
        //// output
        // -c ?
        if (covmat_file_given) {
          verbose && ( ! use_correlation) && std::cerr << "writing covariance matrix" << std::endl;
          verbose &&     use_correlation  && std::cerr << "writing correlation matrix" << std::endl;
          std::string covmat_file = args["cov"].as<std::string>();
          FastPCA::DataFileWriter<double>(covmat_file).write(FastPCA::Matrix<double>(cov));
        }
        // -v ?
        if (eigenvec_file_given) {
          verbose && std::cerr << "solving eigensystem/writing eigenvectors matrix" << std::endl;
          std::string eigenvec_file = args["vec"].as<std::string>();
          FastPCA::DataFileWriter<double>(eigenvec_file).write(cov.eigenvectors());
        }
        // -l ?
        if (eigenval_file_given) {
          verbose && std::cerr << "solving eigensystem/writing eigenvalues matrix" << std::endl;
          std::string eigenval_file = args["val"].as<std::string>();
          FastPCA::DataFileWriter<double>(eigenval_file).write(cov.eigenvalues());
        }
        // -p ?
        if (projection_file_given) {
          std::string projection_file = args["proj"].as<std::string>();
          if (periodic) {
            verbose && std::cerr << "computing projections for periodic data" << std::endl;
            FastPCA::Periodic::calculate_projections(file_input, projection_file, vecs, mem_buf_size, use_correlation, stats);
          } else {
            verbose && std::cerr << "computing projections" << std::endl;
            FastPCA::calculate_projections(file_input, projection_file, vecs, mem_buf_size, use_correlation, stats);
          }
        }
        // -s ?
        if (stats_file_given) {
          verbose && std::cerr << "writing stats" << std::endl;
          FastPCA::DataFileWriter<double>(args["stats"].as<std::string>()).write(stats);
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
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    std::cerr << desc << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


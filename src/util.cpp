
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


#include "util.hpp"
#include "file_io.hpp"

#include <string>
#include <algorithm>

namespace FastPCA {

  namespace {
    std::tuple<std::size_t, std::size_t, std::vector<double>>
    _means(const std::string filename
         , const std::size_t max_chunk_size) {
      DataFileReader<double> input_file(filename, max_chunk_size);
      std::size_t n_cols = input_file.n_cols();
      std::size_t n_rows = 0;
      std::vector<double> means(n_cols);
      Matrix<double> m = std::move(input_file.next_block());
      while (m.n_rows() > 0) {
        n_rows += m.n_rows();
        for (std::size_t i=0; i < m.n_rows(); ++i) {
          for (std::size_t j=0; j < n_cols; ++j) {
            means[j] += m(i,j);
          }
        }
        m = std::move(input_file.next_block());
      }
      for (std::size_t j=0; j < n_cols; ++j) {
        means[j] /= n_rows;
      }
      return std::make_tuple(n_rows, n_cols, means);
    }

    // compute circular means by averaging sines and cosines
    // and resolving the mean angle with the atan2 function.
    // additionally, return number of observations.
    std::tuple<std::size_t, std::size_t, std::vector<double>>
    _circular_means(const std::string filename
                  , const std::size_t max_chunk_size) {
      DataFileReader<double> input_file(filename, max_chunk_size);
      Matrix<double> m = std::move(input_file.next_block());
      FastPCA::deg2rad_inplace(m);
      std::size_t i, j;
      std::size_t nr = m.n_rows();
      std::size_t nc = m.n_cols();
      std::vector<double> means(nc, 0.0);
      std::vector<double> means_sin(nc, 0.0);
      std::vector<double> means_cos(nc, 0.0);
      std::size_t n_rows_total = 0;
      while (nr > 0) {
        for (j=0; j < nc; ++j) {
          for (i=0; i < nr; ++i) {
            means_sin[j] += sin(m(i,j));
            means_cos[j] += cos(m(i,j));
          }
        }
        n_rows_total += nr;
        m = std::move(input_file.next_block());
        FastPCA::deg2rad_inplace(m);
        nr = m.n_rows();
      }
      for (j=0; j < nc; ++j) {
        means[j] = std::atan2(means_sin[j]/n_rows_total, means_cos[j]/n_rows_total);
      }
      return std::make_tuple(n_rows_total, nc, means);
    }

    std::vector<double>
    _sigmas(const std::string filename
          , const std::size_t max_chunk_size
          , std::vector<double> means
          , bool periodic) {
      DataFileReader<double> input_file(filename, max_chunk_size);
      std::size_t n_cols = means.size();
      std::size_t n_rows = 0;
      std::vector<double> sigmas(n_cols, 0.0);
      Matrix<double> m = std::move(input_file.next_block());
      while (m.n_rows() > 0) {
        n_rows += m.n_rows();
        // subtract means
        if (periodic) {
          FastPCA::deg2rad_inplace(m);
          FastPCA::Periodic::shift_matrix_columns_inplace(m, means);
        } else {
          FastPCA::shift_matrix_columns_inplace(m, means);
        }
        // compute variances
        for (std::size_t i=0; i < m.n_rows(); ++i) {
          for (std::size_t j=0; j < n_cols; ++j) {
            sigmas[j] += m(i,j)*m(i,j);
          }
        }
        m = std::move(input_file.next_block());
      }
      // compute sigmas from variances
      for (std::size_t j=0; j < n_cols; ++j) {
        sigmas[j] = sqrt(sigmas[j] / (n_rows-1));
      }
      return sigmas;
    }

    double
    _periodic_shift_to_barrier_deg(double theta, double shift) {
      theta -= (shift + 180.0);
      if (theta < -180.0) {
        return theta + 360.0;
      } else if (theta > 180.0) {
        return theta - 360.0;
      } else {
        return theta;
      }
    }

    unsigned int
    _count_jumps_deg(const Matrix<double>& m, std::size_t i_col, double shift) {
      unsigned int sum = 0;
      for (std::size_t i=0; i < m.n_rows()-1; ++i) {
        double theta1 = _periodic_shift_to_barrier_deg(m(i,i_col), shift);
        double theta2 = _periodic_shift_to_barrier_deg(m(i+1,i_col), shift);
        if (std::abs(theta1 - theta2) > 180.0) {
          ++sum;
        }
      }
      return sum;
    }

    std::size_t
    _n_cols(const std::string filename) {
      return DataFileReader<double>(filename, 1024).n_cols();
    }

    std::vector<std::vector<unsigned int>>
    _shift_hists(const std::string filename
               , std::size_t max_chunk_size
               , std::size_t n_bins
               , std::size_t binwidth) {
      std::size_t n_cols = _n_cols(filename);
      // result
      std::vector<std::vector<unsigned int>>
        hists(n_cols
            , std::vector<unsigned int>(n_bins));
      // compute histograms
      DataFileReader<double> input_file(filename, max_chunk_size/2);
      read_blockwise(input_file
                   , [&hists,binwidth,n_bins](Matrix<double>& m) {
        std::size_t i, j, i_bin;
        #pragma omp parallel for default(none)\
                                 private(j,i,i_bin)\
                                 firstprivate(n_cols,n_bins,binwidth)\
                                 shared(hists,m)
        for (j=0; j < m.n_cols(); ++j) {
          for (i=0; i < m.n_rows(); ++i) {
            for (i_bin=0; i_bin < n_bins; ++i_bin) {
              if (m(i,j) <= -180.0f + (i_bin+1)*binwidth) {
                ++hists[j][i_bin];
                break;
              }
            }
          }
        }
      });
      return hists;
    }

    std::vector<double>
    _shift_barrier(std::vector<double> barriers) {
      // shift not to center, but to barrier
      for (double& b: barriers) {
        b += 180.0;
        if (b > 180.0) {
          b -= 360.0;
        }
      }
      deg2rad_inplace(barriers);
      return barriers;
    }

    std::vector<double>
    _dih_shifts_static(const std::string filename
                     , std::size_t max_chunk_size) {
      max_chunk_size /= 2;
      // get number of columns from file
      std::size_t n_cols = _n_cols(filename);
      // histograms of degree populations
      // (72 bins of 5 degrees width, [-180, 180])
      const std::size_t n_bins = 72;
      const float binwidth = 5.0;
      std::vector<std::vector<unsigned int>> hists =
        _shift_hists(filename
                   , max_chunk_size
                   , n_bins
                   , binwidth);
      // periodic index correction
      auto periodic = [n_bins](int i) -> int {
        if (i < 0) {
          return n_bins + i;
        } else if (i >= n_bins) {
          return i - n_bins;
        } else {
          return i;
        }
      };
      // find barriers per column
      std::vector<double> barriers(n_cols);
      for (unsigned int i_col=0; i_col < n_cols; ++i_col) {
        // bin statistics for comparison
        // (pop of bin and sum(pops) over 3, 5, 7 neighboring bins)
        std::vector<std::pair<unsigned int
                            , std::array<unsigned int, 4>>> bin_stats(n_bins);
        for (int i=0; i < n_bins; ++i) {
          bin_stats[i].first = i;
          // bin itself
          bin_stats[i].second[0] = hists[i_col][i];
          // sum 3 neighbors
          bin_stats[i].second[1] = bin_stats[i].second[0]
                                 + hists[i_col][periodic(i-1)]
                                 + hists[i_col][periodic(i+1)];
          // sum 5 neighbors
          bin_stats[i].second[2] = bin_stats[i].second[1]
                                 + hists[i_col][periodic(i-2)]
                                 + hists[i_col][periodic(i+2)];
          // sum 7 neighbors
          bin_stats[i].second[3] = bin_stats[i].second[2]
                                 + hists[i_col][periodic(i-3)]
                                 + hists[i_col][periodic(i+3)];
        }
        // comparator: a < b ?
        auto bin_stats_comp = [](std::pair<unsigned int
                                         , std::array<unsigned int, 4>> a
                               , std::pair<unsigned int
                                         , std::array<unsigned int, 4>> b) {
          return (a.second[0] < b.second[0])
              || ((a.second[0] == b.second[0])
               && (a.second[1] < b.second[1]))
              || ((a.second[0] == b.second[0])
               && (a.second[1] == b.second[1])
               && (a.second[2] < b.second[2]))
              || ((a.second[0] == b.second[0])
               && (a.second[1] == b.second[1])
               && (a.second[2] == b.second[2])
               && (a.second[3] < b.second[3]));
        };
        unsigned int best_bin = (*std::min_element(bin_stats.begin()
                                                 , bin_stats.end()
                                                 , bin_stats_comp)).first;
        // compute best shift
        barriers[i_col] = -180 + (best_bin * binwidth) + 0.5 * binwidth;
      }
      return _shift_barrier(barriers);
    }

    std::vector<double>
    _dih_shifts_dynamic(const std::string filename
                      , std::size_t max_chunk_size) {
      // get number of columns from file
      std::size_t n_cols;
      {
        DataFileReader<double> input_file(filename, max_chunk_size);
        n_cols = input_file.n_cols();
      }
      // histogram for first guess (72 bins of 5 degrees width, [-180, 180])
      const std::size_t n_bins = 72;
      const float binwidth = 5.0;
      std::vector<std::vector<unsigned int>> hists =
        _shift_hists(filename
                   , max_chunk_size
                   , n_bins
                   , binwidth);
      // compute min-shift candidates from histograms
      const int n_min_bins = 5;
      const int n_values_per_bin = 5;
      const int n_candidates_per_col = n_min_bins * n_values_per_bin;
      std::vector<std::vector<float>>
        candidates(n_cols
                 , std::vector<float>(n_candidates_per_col));
      for (std::size_t j=0; j < n_cols; ++j) {
        // 5 smallest bins
        std::vector<std::size_t> min_bins(5, 0);
        for (std::size_t i_bin=1; i_bin < n_bins; ++i_bin) {
          for (std::size_t k=0; k < n_min_bins; ++k) {
            if (hists[j][i_bin] < hists[j][min_bins[k]]) {
              min_bins[k] = i_bin;
              break;
            }
          }
        }
        // compute degree values of candidates
        // candidates: 5 values for 5 bins = 25 values.
        //   compute by: deg(bin), deg(bin)+1.0, deg(bin)+2.0, ..., deg(bin)+4.0
        //   with deg(bin) degree value of bin (lower boundary).
        for (std::size_t i_min_bin=0; i_min_bin < n_min_bins; ++i_min_bin) {
          for (std::size_t k=0; k < n_values_per_bin; ++k) {
            candidates[j][i_min_bin*n_min_bins + k] =
              min_bins[i_min_bin]*binwidth + k -180.0f;
          }
        }
      }
      // compute ranking for different shifts
      std::vector<std::vector<unsigned int>>
        n_jumps(n_cols
              , std::vector<unsigned int>(n_candidates_per_col));
      {
        DataFileReader<double> input_file(filename, max_chunk_size/2);
        read_blockwise(input_file
                     , [&n_jumps
                      , &candidates
                      , n_cols
                      , n_candidates_per_col] (Matrix<double>& m) {
          std::size_t j;
          std::size_t ic;
          #pragma omp parallel for default(none)\
                                   private(j,ic)\
                                   firstprivate(n_cols,n_candidates_per_col)\
                                   shared(m,candidates,n_jumps)
          for (j=0; j < n_cols; ++j) {
            for (ic=0; ic < n_candidates_per_col; ++ic) {
              n_jumps[j][ic] = _count_jumps_deg(m, j, candidates[j][ic]);
            }
          }
        });
      }
      // shifts: minimal jumps win
      std::vector<double> shifts(n_cols);
      for (std::size_t j=0; j < n_cols; ++j) {
        unsigned int jumps_min = n_jumps[j][0];
        std::size_t i_min = 0;
        for (std::size_t i=1; i < n_candidates_per_col; ++i) {
          if (n_jumps[j][i] < jumps_min) {
            i_min = i;
            jumps_min = n_jumps[j][i];
          }
        }
        shifts[j] = candidates[j][i_min];
      }
      // put cut points on periodic barriers
      return _shift_barrier(shifts);
    }

    Matrix<double>
    _stats(const std::string filename
         , const std::size_t max_chunk_size
         , bool periodic
         , bool dynamic_shift) {
      std::vector<double> means;
      std::size_t n_rows, n_cols;
      // means
      if (periodic) {
        std::tie(n_rows, n_cols, means) = _circular_means(filename
                                                        , max_chunk_size);
      } else {
        std::tie(n_rows, n_cols, means) = _means(filename
                                               , max_chunk_size);
      }
      // sigmas
      std::vector<double> sigmas = _sigmas(filename
                                         , max_chunk_size
                                         , means
                                         , periodic);
      // shifts
      std::vector<double> shifts;
      if (periodic && dynamic_shift) {
        shifts = _dih_shifts_dynamic(filename
                                   , max_chunk_size);
      } else if (periodic && ( ! dynamic_shift)) {
        shifts = _dih_shifts_static(filename
                                  , max_chunk_size);
      }
      int n_cols_out;
      if (periodic) {
        n_cols_out = 3;
      } else {
        n_cols_out = 2;
      }
      Matrix<double> stats(n_cols, n_cols_out);
      for (std::size_t i=0; i < n_cols; ++i) {
        stats(i, 0) = means[i];
        stats(i, 1) = sigmas[i];
        if (periodic) {
          stats(i, 2) = shifts[i];
        }
      }
      return stats;
    }

  } // end local namespace


  bool is_comment_or_empty(std::string line) {
    std::size_t pos = line.find_first_not_of(" ");
    if (pos == std::string::npos) {
      // empty string
      return true;
    } else if (line[pos] == '#' or line[pos] == '@') {
      // comment
      return true;
    } else {
      return false;
    }
  }

  Matrix<double>
  stats(const std::string filename
      , const std::size_t max_chunk_size) {
    return _stats(filename
                , max_chunk_size
                , false
                , false);
  }


  namespace Periodic {
    double
    distance(double theta1, double theta2) {
      double abs_diff = std::abs(theta1 - theta2);
      if (abs_diff <= M_PI) {
        return abs_diff;
      } else {
        return abs_diff - (2*M_PI);
      }
    }

    Matrix<double>
    stats(const std::string filename
        , const std::size_t max_chunk_size
        , const bool dynamic_shift) {
      return _stats(filename, max_chunk_size, true, dynamic_shift);
    }

  } // end namespace FastPCA::Periodic
} // end namespace FastPCA


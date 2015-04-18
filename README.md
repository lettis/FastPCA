# About #

FastCA is a PCA-calculator programmed in C++(11).
Computation is parallelized with OpenMP.

For fast matrix diagonalization, LAPACK is used (and needed, of course).

The project includes the 'xdrfile' library of GROMACS. Thus, you can
use data files written as ASCII data as well as .xtc-trajectories.

For bug-reports, write to

   sittel@lettis.net
or
   florian.sittel@physik.uni-freiburg.de

Happy Computing.



# Compilation #

Create a build-directory in the project root and change into
that directory:

 # mkdir build
 
 # cd build

Run cmake, based on the underlying project:

 # cmake ..

Hopefully, everything went right.
If not, carefully read the error messages.
Typical errors are missing dependencies...

If everything is o.k., run make (on multicore machines, use '-j' to parallelize
compilation, e.g. 'make -j 4' for up to four parallel jobs):

 # make

Now, you should find the 'fastca' binary in the 'src' folder.


## Requirements ##
  * LAPACK
  * Boost (program_options), min. version 1.49
  * cmake, min. version 2.8
  * g++


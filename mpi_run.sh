#!/bin/bash

if [ "$1" == "100D" ]; then
  echo "Running test with input100D"
  make KMEANS_mpi
  mpirun -n 4 ./KMEANS_mpi test_files/input100D.inp 40 100 1 0.0001 output_files/mpi/output100D.txt
elif [ "$1" == "100D2" ]; then
  echo "Running test with input100D2"
  make KMEANS_mpi
  mpirun -n 4 ./KMEANS_mpi test_files/input100D2.inp 40 100 1 0.0001 output_files/mpi/output100D2.txt
fi

make clean


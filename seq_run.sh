#!/bin/bash

if [ "$1" == "100D" ]; then
  echo "Running test with input100D"
  make KMEANS_seq
  ./KMEANS_seq test_files/input100D.inp 40 100 1 0.0001 output_files/seq/output100D.txt
elif [ "$1" == "100D2" ]; then
  echo "Running test with input100D2"
  make KMEANS_seq
  ./KMEANS_seq test_files/input100D2.inp 40 100 1 0.0001 output_files/seq/output100D2.txt
elif [ "$1" == "2D" ]; then
  echo "Runnin test with input2D"
  make KMEANS_seq
  ./KMEANS_seq test_files/input2D.inp 40 100 1 0.0001 output_files/seq/output2D.txt
elif [ "$1" == "2D2" ]; then
  echo "Running test with input2D2"
  make KMEANS_seq
  ./KMEANS_seq test_files/input2D2.inp 40 100 1 0.0001 output_files/seq/output2D2.txt
elif [ "$1" == "10D" ]; then
  echo "Running test with input10D"
  make KMEANS_seq
  ./KMEANS_seq test_files/input10D.inp 40 100 1 0.0001 output_files/seq/output10D.txt
elif [ "$1" == "20D" ]; then
  echo "Running test with input20D"
  make KMEANS_seq
  ./KMEANS_seq test_files/input20D.inp 40 100 1 0.0001 output_files/seq/output20D.txt
fi

make clean


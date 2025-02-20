#!/bin/bash

if [ "$1" == "100D" ]; then
    echo "Running test with input100D"
    make KMEANS_seq
    ./KMEANS_seq test_files/input100D.inp 40 100 1 0.0001 output_files/seq/output100D.txt
elif [ "$1" == "1002" ]; then
    echo "Running test with input100D2"
    make KMEANS_seq
    ./KMEANS_seq test_files/input100D2.inp 40 100 1 0.0001 output_files/seq/output100D2.txt
fi
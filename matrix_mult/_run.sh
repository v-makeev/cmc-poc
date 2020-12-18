#!/bin/bash

sudo docker run -v $PWD:/sandbox:Z abouteiller/mpi-ft-ulfm mpirun --oversubscribe -mca btl tcp,self "$@"

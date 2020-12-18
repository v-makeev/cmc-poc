#include <mpi/mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int size, rank;
double num, tmp, sum_to_i, partial_sum, ans;
MPI_Status status;
MPI_Request reqs[2];
MPI_Status st[2];

double GetUniformRand(double a, double b) {
    return (double) rand() / RAND_MAX * (b - a) + a;
}

void send(int from, int to) {
    if (rank == from) MPI_Send(&partial_sum, 1, MPI_DOUBLE, to, 0, MPI_COMM_WORLD);
    if (rank == to) MPI_Recv(&tmp, 1, MPI_DOUBLE, from, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
}

void send_par(int from, int to, MPI_Request *req) {
    if (rank == from) MPI_Isend(&partial_sum, 1, MPI_DOUBLE, to, 0, MPI_COMM_WORLD, req);
    if (rank == to) MPI_Irecv(&tmp, 1, MPI_DOUBLE, from, 0, MPI_COMM_WORLD, req);
}

int main(int argc, char *argv[]) {

    setvbuf(stdout, NULL, _IOLBF, BUFSIZ);
    setvbuf(stderr, NULL, _IOLBF, BUFSIZ);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(time(NULL) * rank);

    num = GetUniformRand(0, 10);
    partial_sum = num;
    sum_to_i = num;

    MPI_Barrier(MPI_COMM_WORLD);

    // 1 шаг
    switch (rank % 4) {
        case 0:
            MPI_Send(&partial_sum, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            break;
        case 1:
            MPI_Recv(&tmp, 1, MPI_DOUBLE, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            sum_to_i += tmp;
            partial_sum += tmp;
            break;
        case 2:
            MPI_Recv(&tmp, 1, MPI_DOUBLE, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            partial_sum += tmp;
            break;
        case 3:
            MPI_Send(&partial_sum, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            break;
    }

    switch (rank %4) {
        case 1:
            MPI_Isend(&partial_sum, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Irecv(&tmp, 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &reqs[1]);
            MPI_Waitall(2, reqs, st);
            partial_sum += tmp;
            break;
        case 2:
            MPI_Isend(&partial_sum, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Irecv(&tmp, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &reqs[1]);
            MPI_Waitall(2, reqs, st);
            sum_to_i += tmp;
            partial_sum += tmp;
            break;

    }

    // 3 шаг
    switch (rank % 4) {
        case 1:
            MPI_Send(&partial_sum, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            break;
        case 0:
            MPI_Recv(&tmp, 1, MPI_DOUBLE, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            partial_sum = tmp;
            break;
        case 3:
            MPI_Recv(&tmp, 1, MPI_DOUBLE, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            partial_sum = tmp;
            sum_to_i = partial_sum;
            break;
        case 2:
            MPI_Send(&partial_sum, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            break;
    }

    // шаги 4-6
    if (rank / 4 != 0) {
        MPI_Recv(&tmp, 1, MPI_DOUBLE, rank - 4, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        sum_to_i += tmp;
        partial_sum += tmp;
    }
    if (rank / 4 != 3) {
        MPI_Send(&partial_sum, 1, MPI_DOUBLE, rank + 4, 0, MPI_COMM_WORLD);
    }

    printf("proc #%d, num = %.2f, sum = %.2f\n", rank, num, sum_to_i);
    MPI_Barrier(MPI_COMM_WORLD);
}
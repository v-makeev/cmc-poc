#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
//#include <mpi/mpi.h>
#include "mpi-ext.h"
#include <signal.h>
#include <unistd.h>
#include <stdarg.h>
#include <string.h>

int proc_num, myrank;
int dummy_int;

void proc_print(char *text, ...) {
    char indent[myrank*8 + 1];
    for (int i = 0; i < myrank*8; ++i) {
        if (i % 8 == 0) {
            indent[i] = '|';
        } else {
            indent[i] = ' ';
        }
    }
    indent[myrank*8] = '\0';

    printf("%s#%d: ",indent, myrank);
    va_list args;
    va_start(args, text);
    vprintf(text, args);
    va_end(args);
}

static
void init_array(int ni, int nj, int nk,
                float *alpha,
                float *beta,
                float C[ni][nj],
                float A[ni][nk],
                float B[nk][nj]) {
    int i, j;
    *alpha = 1.5;
    *beta = 1.2;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++)
            C[i][j] = (float) ((i * j + 1) % ni) / ni;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nk; j++)
            A[i][j] = (float) (i * (j + 1) % nk) / nk;
    for (i = 0; i < nk; i++)
        for (j = 0; j < nj; j++)
            B[i][j] = (float) (i * (j + 2) % nj) / nj;
}

static
void kernel_gemm(int ni, int nj, int nk,
                 float alpha,
                 float beta,
                 float C[ni][nj],
                 float A[ni][nk],
                 float B[nk][nj],
                 int i_1,
                 int i_2,
                 FILE *f) {
    int i, j, k;

    for (i = i_1; i < i_2; i++) {
//        if ((myrank == 1 || myrank == 5 || myrank == 7) && i == i_1 + 1) {
//            proc_print("failed on iteration %d!\n", i - i_1);
//            raise(SIGKILL);
//        }
        for (j = 0; j < nj; j++) {
            C[i][j] *= beta;
            for (k = 0; k < nk; k++) {
                C[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
        fwrite(C[i], sizeof(float), nj, f);
        fflush(f);
    }
}

void calculate_bounds(int ni, int rank, int *i_min, int *i_max) {
    *i_min = ni / proc_num * rank;
    *i_max = ni / proc_num * (rank + 1);
    if (rank < ni % proc_num) {
        *i_min += rank;
        *i_max += rank + 1;
    } else {
        *i_min += ni % proc_num;
        *i_max += ni % proc_num;
    }
}

FILE *read_backup(int ni, int nj, int old_rank, float C[ni][nj], int *i_min, int *i_max, int *read_rows) {
    *read_rows = 0;
    char str[20];
    sprintf(str, "%d_res", old_rank);
    proc_print("opening file %s\n", str);
    FILE *f = fopen(str, "r");

    if (f == NULL) {
        calculate_bounds(ni, old_rank, i_min, i_max);
        return fopen(str, "a");
    }

    unsigned read_i_min = fread(i_min, sizeof(*i_min), 1, f);
    unsigned read_i_max = fread(i_max, sizeof(*i_max), 1, f);

    if (read_i_max == 0 || read_i_min == 0) {
        calculate_bounds(ni, old_rank, i_min, i_max);
    }

    float buf[nj];
    int count = *i_min;
    while (1) {
        unsigned elems_read = fread(buf, sizeof(float), nj, f);
        if (elems_read != nj) {
            break;
        }
        for (int j = 0; j < nj; ++j) {
            C[count][j] = buf[j];
        }
        count++;
        (*read_rows)++;
    }
    *i_min = count;

    fclose(f);
    return fopen(str, "a");
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    // размеры матриц
    int ni = atoi(argv[1]);
    int nj = atoi(argv[2]);
    int nk = atoi(argv[3]);

    // последние argv[4] процессов используются как вспомогательные
    // процесс с номером proc_num - argv[4] является мастером и управляет процессами в случае сбоя
    int backup_proc_num = atoi(argv[4]);
    proc_num -= backup_proc_num + 1;
    int main_proc_rank = proc_num;

    if (myrank == 0) {
        printf("There are %d processes total\n", proc_num + backup_proc_num + 1);
        printf("\tProc #0-#%d are workes\n", proc_num - 1);
        printf("\tProc #%d is master\n", main_proc_rank);
        printf("\tProc #%d-#%d are backup\n", proc_num + 1, proc_num + backup_proc_num);
        printf("i = %d, j = %d, k = %d\n\n", ni, nj, nk);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Request reqs[proc_num];

    // мастер процесс
    if (myrank == main_proc_rank) {
        // ранги рабочих процессов.
        int worker_ranks[proc_num];
        // ранги вспомогательных процессов
        int backup_ranks[backup_proc_num];
        // индекс первого свободного процесса в массиве backup_ranks (который еще не использован)
        int backupIdx = 0;

        for (int i = 0; i < proc_num; ++i) {
            worker_ranks[i] = i;
        }
        for (int i = 0; i < backup_proc_num; ++i) {
            backup_ranks[i] = proc_num + i + 1;
        }

        // посылаем каждому рабочему процессу произвольное сообщение
        for (int i = 0; i < proc_num; ++i) {
            MPI_Irecv(&dummy_int, 1, MPI_INT, worker_ranks[i], 1234, MPI_COMM_WORLD, &reqs[i]);
        }

        while (1) {
            int completed_idx;
            // ждем завершения какого-то рабочего процесса
            int s = MPI_Waitany(proc_num, reqs, &completed_idx, MPI_STATUS_IGNORE);
            proc_print("received status %d and index %d from wait\n", s, completed_idx);

            // ждать больше некого, все процессы завершились
            if (completed_idx < 0) {
                break;
            }
            // с процессом что-то не так
            if (s != MPI_SUCCESS) {
                proc_print("found issues in process #%d\n", worker_ranks[completed_idx]);
                // если еще остались вспомогательные процессы, выделяем
                if (backupIdx < backup_proc_num) {
                    MPI_Request_free(&reqs[completed_idx]);
                    proc_print("sending backup data to process #%d\n", backup_ranks[backupIdx]);
                    // этот вспомогательный процесс заменит упавший рабочий
                    // обновляем ранги соответствующим образом
                    worker_ranks[completed_idx] = backup_ranks[backupIdx];
                    backupIdx++;
                    // будим вспомогательный процесс
                    // отправляя ему его индекс в массиве
                    MPI_Send(&completed_idx, 1, MPI_INT, backup_ranks[backupIdx - 1], 1, MPI_COMM_WORLD);
                    // добавляем этот процесс в список ожидания
                    MPI_Irecv(&dummy_int, 1, MPI_INT, worker_ranks[completed_idx], 1234, MPI_COMM_WORLD, &reqs[completed_idx]);
                } else {
                    proc_print("no more backup procs left :(\n");
                }
            }
        }

        // будим все процессы, которые не понадобились
        for (int i = backupIdx; i < backup_proc_num; ++i) {
            int val = -1;
            MPI_Send(&val, 1, MPI_INT, backup_ranks[i], 1, MPI_COMM_WORLD);
        }

        proc_print("master finishing\n");
        MPI_Finalize();
        return 0;
    }

    float alpha;
    float beta;
    float (*C)[ni][nj];
    C = (float (*)[ni][nj]) malloc((ni) * (nj) * sizeof(float));
    float (*A)[ni][nk];
    A = (float (*)[ni][nk]) malloc((ni) * (nk) * sizeof(float));
    float (*B)[nk][nj];
    B = (float (*)[nk][nj]) malloc((nk) * (nj) * sizeof(float));

    init_array(ni, nj, nk, &alpha, &beta,
               *C,
               *A,
               *B);

    // рабочий процесс
    if (myrank < proc_num) {
        int i_min, i_max;
        calculate_bounds(ni, myrank, &i_min, &i_max);

        proc_print("working in range [%d; %d)\n", i_min, i_max);

        // получаем имя файла, в который будут сбрасываться результаты во время работы
        char str[20];
        sprintf(str, "%d_res", myrank);
        FILE *f = fopen(str, "w");
        // пишем в начало файла индексы, которые считает процесс
        fwrite(&i_min, sizeof(i_min), 1, f);
        fwrite(&i_max, sizeof(i_max), 1, f);

        // начинаем вычислять, на каждом шаге записывая результаты в файл
        kernel_gemm(ni, nj, nk,
                    alpha, beta,
                    *C,
                    *A,
                    *B,
                    i_min,
                    i_max,
                    f);
        fclose(f);

        // отправляем сигнал мастеру о завершении работы
        MPI_Isend(&dummy_int, 1, MPI_INT, proc_num, 1234, MPI_COMM_WORLD, &reqs[myrank]);
    } else { // вспомогательный процесс
        proc_print("waiting as backup\n", myrank);
        // ждем, пока главный процесс пришлет номер файла с результатами
        int replace_num;
        MPI_Recv(&replace_num, 1, MPI_INT, main_proc_rank, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        proc_print("received %d\n", replace_num);
        // процесс не понадобился, завершаемся
        if (replace_num < 0) {
            proc_print("finished\n");
            MPI_Finalize();
            return 0;
        }
        // считываем информацию из файла и продолжаем считать
        int i_min, i_max, read_rows;
        FILE *f = read_backup(ni, nj, replace_num, *C, &i_min, &i_max, &read_rows);
        proc_print("read %d rows, calculating rows from %d to %d\n", read_rows, i_min, i_max - 1);
        kernel_gemm(ni, nj, nk,
                    alpha, beta,
                    *C,
                    *A,
                    *B,
                    i_min,
                    i_max,
                    f);
        fclose(f);

        // отправляем сигнал мастеру о завершении работы
        MPI_Isend(&dummy_int, 1, MPI_INT, proc_num, 1234, MPI_COMM_WORLD, &reqs[replace_num]);
    }

    free((void *) C);
    free((void *) A);
    free((void *) B);

    proc_print("finished\n");
    MPI_Finalize();
    return 0;
}
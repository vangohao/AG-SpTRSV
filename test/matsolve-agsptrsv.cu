// ref:
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/spsv_csr/spsv_csr_example.c

#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <string>

#include "unisolver/ArrayUtils.hpp"
#include "unisolver/JsonUtils.hpp"

#include "AG-SpTRSV.h"

#include "utils.h"

#include "YYSpTRSV.h"

#include "spts_syncfree_cuda.h"

using namespace uni;

#define VALUE_TYPE double
#define VALUE_SIZE 8

#define REPEAT_TIME 11
#define WARM_UP 1

#define ag_duration(a, b) \
    (1.0 * (b.tv_usec - a.tv_usec + (b.tv_sec - a.tv_sec) * 1.0e6))

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess) {                                   \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            while (1);                                                 \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS) {                           \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            while (1);                                                     \
        }                                                                  \
    }

using cusp_int = int;
#define my_CUSPARSE_INDEX CUSPARSE_INDEX_32I

#define MAX_DOF_TEST 8

struct benchmark_record {
    double total_time = 0;
    long flops = 0;
    long bytes = 0;
    long count = 0;
};

benchmark_record benchmark_record_map_lower[MAX_DOF_TEST];

void RunBenchmarkLowerWithCusparse(Json json, int Dof, int stencil_type,
                                   int stencil_width) {
    constexpr int Dim = 3;

    std::string dof_str = std::to_string(Dof);
    cusp_int M = json[dof_str]["M"].get<cusp_int>();
    cusp_int N = json[dof_str]["N"].get<cusp_int>();
    cusp_int P = json[dof_str]["P"].get<cusp_int>();

    std::vector<std::array<cusp_int, Dim>> stencil_points;
    if (stencil_type == 0) {
        for (int d = Dim - 1; d >= 0; d--) {
            for (int j = stencil_width; j > 0; j--) {
                std::array<cusp_int, Dim> pt = {0, 0, 0};
                pt[d] = -j;
                stencil_points.push_back(pt);
            }
        }
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, 0});
    } else if (stencil_type == 1) {
        NestedLoop(
            constant_array<cusp_int, Dim>(-stencil_width),
            constant_array<cusp_int, Dim>(2 * stencil_width + 1), [&](auto pt) {
                cusp_int cnt = CartToFlat(
                    pt + stencil_width,
                    constant_array<cusp_int, Dim>(2 * stencil_width + 1));
                if (cnt < (myPow(2 * stencil_width + 1, Dim) / 2)) {
                    stencil_points.push_back(pt);
                }
            });
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, 0});
    } else {
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, -1});
        stencil_points.push_back(std::array<cusp_int, Dim>{1, 0, -1});
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 1, -1});
        stencil_points.push_back(std::array<cusp_int, Dim>{0, -1, 0});
        stencil_points.push_back(std::array<cusp_int, Dim>{1, -1, 0});
        stencil_points.push_back(std::array<cusp_int, Dim>{-1, 0, 0});
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, 0});
    }

    // Host problem definition
    cusp_int A_num_rows = M * N * P * Dof;
    cusp_int A_nnz = 0;
    std::vector<cusp_int> hA_csrOffsets;
    std::vector<cusp_int> hA_columns;
    std::vector<double> hA_values;
    std::vector<double> hX;
    std::vector<double> hY;
    std::vector<double> hY_result;
    // 注意这里求解的是A* Y = X, 所以这里的Y是输出, X是输入

    // set A & hX
    NestedLoop(
        std::array<cusp_int, Dim>{}, std::array<cusp_int, Dim>{M, N, P},
        [&](auto loc) {
            for (int d = 0; d < Dof; d++) {
                hA_csrOffsets.push_back(A_nnz);
                cusp_int cnt = 0;
                for (auto pt : stencil_points) {
                    if (in_range(loc + pt, std::array<cusp_int, Dim>{},
                                 std::array<cusp_int, Dim>{M, N, P} - 1)) {
                        for (int k = 0; k < Dof; k++) {
                            if (pt != std::array<cusp_int, Dim>{0, 0, 0} ||
                                k == d) {
                                hA_columns.push_back(
                                    CartToFlat(
                                        loc + pt,
                                        std::array<cusp_int, Dim>{M, N, P}) *
                                        Dof +
                                    k);
                                hA_values.push_back(1.);
                                A_nnz++;
                                cnt++;
                            }
                        }
                    }
                }
                hX.push_back(cnt);
            }
        });
    hA_csrOffsets.push_back(A_nnz);

    std::cout << "A_nnz = " << A_nnz << "\n";

    // set hY
    hY.resize(A_num_rows);
    hY_result.resize(A_num_rows);
    for (cusp_int i = 0; i < A_num_rows; i++) hY_result[i] = 1.0;

    //--------------------------------------------------------------------------

    /* !!!!!! start computing SpTRSV !!!!!!!! */

    struct timeval tv_begin, tv_end;

    gettimeofday(&tv_begin, NULL);

    PREPROCESSING_STRATEGY ps = ROW_BLOCK;
    SCHEDULE_STRATEGY strategy = SIMPLE;
    int rb = 1;

    // int graph_reorder = 0;

    ptr_handler handler;

    // if (graph_reorder) {
    //     printf("Begin reordering\n");

    //     handler =
    //         SpTRSV_preprocessing(A_num_rows, A_nnz, hA_csrOffsets.data(),
    //         hA_columns.data(), ROW_BLOCK, 1);

    //     graph_reorder_with_level(handler);

    //     int permutation[A_num_rows];

    //     matrix_reorder(handler, permutation, hA_csrOffsets.data(),
    //     hA_columns.data(), hA_values.data());

    //     graph_finalize(handler);
    // }

    int flag;
    float sptrsv_time = 0;

    flag = 1;

    // matrix examples
    // atmosmodd
    // anaparas paras = anaparas(1024, 4, ROW_BLOCK, 1, LEVEL_WISE, ONE_LEVEL,
    // SIMPLE, RG_SIMPLE);

    // Wiki-Talk
    // anaparas paras = anaparas(64, 1, ROW_BLOCK, 1, LEVEL_WISE, ONE_LEVEL,
    // WORKLOAD_BALANCE, RG_SIMPLE);

    // cant
    // anaparas paras = anaparas(64, 2, ROW_BLOCK, 1, LEVEL_WISE, ONE_LEVEL,
    // SIMPLE2, RG_SIMPLE);

    // delaunay_n23
    // anaparas paras = anaparas(1024, 4, ROW_BLOCK_AVG, 2, LEVEL_WISE,
    // THRESH_LEVEL, SIMPLE, RG_SIMPLE);

    // delaunay_n13
    // anaparas paras = anaparas(64, 4, ROW_BLOCK_AVG, 1, LEVEL_WISE,
    // THRESH_LEVEL, WORKLOAD_BALANCE, RG_SIMPLE);

    // webbase-1M
    // anaparas paras = anaparas(1024, 1, ROW_BLOCK_THRESH, 32, LEVEL_WISE,
    // ONE_LEVEL, SIMPLE, RG_SIMPLE);

    // my 384**3 d3n4 stencilstar with diag
    // tbs 1024 sws 4 ps 0 8 0 ss 1 0 1
    // 前面都按照顺序，最后三个的顺序是反的
    // anaparas paras = anaparas(1024, 4, ROW_BLOCK, 8, LEVEL_WISE, ONE_LEVEL,
    //                           SIMPLE, RG_BALANCE);

    // my 512**3 d3n4 stencilstar with diag
    // tbs 1024 sws 4 ps 0 16 0 ss 0 0 1
    anaparas paras = anaparas(1024, 4, ROW_BLOCK, 16, LEVEL_WISE, ONE_LEVEL,
                              SIMPLE, RG_SIMPLE);

    // my 320**3 d3n14 stencilbox with diag
    // tbs 1024 sws 4 ps 0 4 0 ss 0 1 1
    // anaparas paras = anaparas(1024, 4, ROW_BLOCK, 4, LEVEL_WISE, ONE_LEVEL,
    // SIMPLE2, RG_SIMPLE);

    // my 384**3 d3n7 stencilstar width = 2 with diag
    // tbs 1024 sws 1 ps 1 16 0 ss 0 0 1
    // anaparas paras = anaparas(1024, 1, ROW_BLOCK_THRESH, 16, LEVEL_WISE,
    // ONE_LEVEL, SIMPLE, RG_SIMPLE);

    // my 384**3 stencilstar width = 1 fill in level =1 with diag
    // tbs 1024 sws 4 ps 0 8 0 ss 0 0 1
    // anaparas paras = anaparas(1024, 4, ROW_BLOCK, 8, LEVEL_WISE, ONE_LEVEL,
    // SIMPLE, RG_SIMPLE);

    show_paras(paras);

    ptr_anainfo ana = new anainfo(A_num_rows);
    SpTRSV_preprocessing_new(A_num_rows, A_nnz, hA_csrOffsets.data(),
                             hA_columns.data(), ana, paras);

    gettimeofday(&tv_end, NULL);

    printf("Preprocessing time: %.2f us\n", ag_duration(tv_begin, tv_end));

    // copy matrix and vector from CPU to GPU memory
    int *csrRowPtr_d, *csrColIdx_d;
    VALUE_TYPE *csrValue_d, *b_d, *x_d;
    cudaMalloc(&csrRowPtr_d, sizeof(int) * (A_num_rows + 1));
    cudaMemcpy(csrRowPtr_d, hA_csrOffsets.data(),
               sizeof(int) * (A_num_rows + 1), cudaMemcpyHostToDevice);
    cudaMalloc(&csrColIdx_d, sizeof(int) * A_nnz);
    cudaMemcpy(csrColIdx_d, hA_columns.data(), sizeof(int) * A_nnz,
               cudaMemcpyHostToDevice);
    cudaMalloc(&csrValue_d, sizeof(VALUE_TYPE) * A_nnz);
    cudaMemcpy(csrValue_d, hA_values.data(), sizeof(VALUE_TYPE) * A_nnz,
               cudaMemcpyHostToDevice);
    cudaMalloc(&b_d, sizeof(VALUE_TYPE) * A_num_rows);
    cudaMemcpy(b_d, hX.data(), sizeof(VALUE_TYPE) * A_num_rows,
               cudaMemcpyHostToDevice);
    cudaMalloc(&x_d, sizeof(VALUE_TYPE) * A_num_rows);
    cudaMemset(x_d, 0, sizeof(VALUE_TYPE) * A_num_rows);

    for (int i = 0; i < REPEAT_TIME; i++) {
        cudaMemset(ana->get_value, 0, sizeof(int) * A_num_rows);
        cudaMemset(x_d, 0, sizeof(VALUE_TYPE) * A_num_rows);

        cudaDeviceSynchronize();

        gettimeofday(&tv_begin, NULL);

        SpTRSV_executor_variant(ana, paras, csrRowPtr_d, csrColIdx_d,
                                csrValue_d, b_d, x_d);
        cudaDeviceSynchronize();

        gettimeofday(&tv_end, NULL);

        if (i >= WARM_UP) sptrsv_time += ag_duration(tv_begin, tv_end);
    }

    sptrsv_time *= 1e-6;

    cudaMemcpy(hY.data(), x_d, sizeof(VALUE_TYPE) * A_num_rows,
               cudaMemcpyDeviceToHost);

    //--------------------------------------------------------------------------

    long readBytes = (sizeof(cusp_int) + sizeof(double)) * A_nnz +
                     sizeof(cusp_int) * A_num_rows +
                     sizeof(double) * A_num_rows;
    long writeBytes = sizeof(double) * A_num_rows;

    benchmark_record_map_lower[Dof - 1] = {
        sptrsv_time, 2L * A_nnz * (REPEAT_TIME - WARM_UP),
        (readBytes + writeBytes) * (REPEAT_TIME - WARM_UP),
        (REPEAT_TIME - WARM_UP)};
    std::cout << "agsptrsv (10 runs) LowerTime(ms): " << sptrsv_time
              << ", Gflops: "
              << (2L * A_nnz * (REPEAT_TIME - WARM_UP) / sptrsv_time) * 1e-9
              << ", Bandwidth="
              << ((readBytes + writeBytes) * (REPEAT_TIME - WARM_UP) /
                  sptrsv_time) *
                     1e-9
              << std::endl;

    //--------------------------------------------------------------------------
    // device result check

    int correct = 1;
    for (cusp_int i = 0; i < A_num_rows; i++) {
        if (hY[i] !=
            hY_result[i]) {  // direct doubleing point comparison is not
            correct = 0;     // reliable
            // break;
            std::cout << "i = " << i << ", hY[i] = " << hY[i]
                      << ", hY_result[i] = " << hY_result[i] << std::endl;
        }
    }
    if (correct)
        printf("agsptrsv test PASSED\n");
    else
        printf("agsptrsv test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    // log::FunctionEnd(0, 0, 0);
    cudaFree(csrRowPtr_d);
    cudaFree(csrColIdx_d);
    cudaFree(csrValue_d);
    cudaFree(x_d);
    cudaFree(b_d);
}

int main(int argc, char **argv) {
    Json json = LoadJsonFromFile("matsolve-csrgen.json");
    std::string problems[] = {"stencilstar", "stencilbox", "stencilstarfill1"};
    bool if_output = json["output"];
    for (int i = 0; i < 3; i++) {
        for (int stencil_width_0 = 0; stencil_width_0 < (i == 0 ? 2 : 1);
             stencil_width_0++) {
            int stencil_width = stencil_width_0 + 1;
            std::string problem = problems[i];

            std::ofstream of;
            if (if_output) {
                of.open(std::string{"results/matsolve-agsptrsv-"} + problem +
                        "-stencilwidth" + std::to_string(stencil_width) +
                        ".out");
            } else {
                of.open("/dev/null");
            }
            for (int dof = 0; dof < MAX_DOF_TEST; dof++) {
                of << problem << ", width=" << stencil_width
                   << ", dof=" << dof + 1 << std::endl;
                RunBenchmarkLowerWithCusparse(
                    json[problem + std::to_string(stencil_width)], dof + 1, i,
                    stencil_width);
                of << "Lower:";
                double total_time = benchmark_record_map_lower[dof].total_time;
                double total_flops_time =
                    static_cast<double>(benchmark_record_map_lower[dof].flops) /
                    total_time;
                double total_bytes_time =
                    static_cast<double>(benchmark_record_map_lower[dof].bytes) /
                    total_time;

                of << dof + 1 << "," << total_time << ","
                   << total_flops_time * 1e-9 << "," << total_bytes_time * 1e-9
                   << std::endl;
            }
            of.close();
        }
    }
    return 0;
}

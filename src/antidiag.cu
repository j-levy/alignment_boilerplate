
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX4(a,b,c,d) MAX(MAX(a,b),MAX(c,d))

#define DEBUG_PRINT (0)

__global__ void sw_antidiag(int* score, 
                            int2* final_pos, 
                            int* H_diag_minus2, 
                            int* H_diag_minus1, 
                            int* E_diag_minus1, 
                            int* F_diag_minus1,
                            char* query_seq,
                            char* target_seq,
                            const int query_length,
                            const int target_length,
                            const int match_,
                            const int mismatch_,
                            const int gapOpen_,
                            const int gapExtend_
                           )
{

    const int diagonalsNumber = query_length + target_length;
    const int diagonalSize = MIN(query_length + 1, target_length + 1);
    int diagonalIndex = 0;
    
    __shared__ int max_score;
    __shared__ int max_pos_col, max_pos_row;
    max_score = 0;

    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex == 0)
        max_score = -1;

    int E = 0, H = 0, F = 0;
    int matchScore = 0;
    int current_row, current_col;
    int rowCode, colCode;
    int norm_index;

    // diagonalIndex -1 is the corner cell where there is no score, and is not computed.
    // diagonalIndex  0 is the first, 2-cell diagonal.
    for (diagonalIndex = 0; diagonalIndex < diagonalsNumber; diagonalIndex++)
    {
        // current_row / current_col goes from -1 to length-1, and is an index to the sequence (0->length-1 included)
        current_row = query_length - 1 - threadIndex;
        current_col = diagonalIndex - current_row - 1;
        norm_index = threadIndex;

        if (current_row < 0 || current_row > query_length - 1)
            rowCode = -1;
        else
            rowCode = query_seq[current_row];

        if (current_col < 0 || current_col > target_length - 1)
            colCode = -2;
        else
            colCode = target_seq[current_col];
        

        const int E_1 = E_diag_minus1[norm_index] - gapExtend_;
        const int F_1 = F_diag_minus1[norm_index + 1] - gapExtend_;
        const int E_2 = H_diag_minus1[norm_index] - gapOpen_;
        const int F_2 = H_diag_minus1[norm_index + 1] - gapOpen_;
        E = current_col >= 0 && current_col < target_length ? MAX(E_1, E_2) : 0;
        F = current_row >= 0 && current_row < query_length ? MAX(F_1, F_2) : 0;
        
        matchScore = (rowCode == colCode) ? match_ : mismatch_;
        H = MAX4(H_diag_minus2[norm_index + 1] + matchScore, E, F, 0);

        if (DEBUG_PRINT > 20){
            if (current_row >= -1)
                printf("diag:%d\tth:%d\t(norm:%d)\t(%d, %d)\t%c-%c=>%d\tH=%d\n", 
                    diagonalIndex, threadIndex, 
                    norm_index,
                    current_col, current_row,
                    rowCode+65, colCode+65, matchScore, H);

            if(threadIndex == 0)
                printf("-------------------------------\n");
        }

        __syncthreads();

        E_diag_minus1[norm_index] = E;
        F_diag_minus1[norm_index] = F;

        H_diag_minus2[norm_index] = H_diag_minus1[norm_index];
        H_diag_minus1[norm_index] = H;
        if(DEBUG_PRINT > 12) {
            __syncthreads();
            if(threadIndex == 0)
            {
                printf("H_diag_minus2 = ");
                for (int k = 0; k < diagonalSize; k++)
                    printf("%d ", H_diag_minus2[k]);
                printf("\nH_diag_minus1 = ");
                for (int k = 0; k < diagonalSize; k++)
                    printf("%d ", H_diag_minus1[k]);
            }
        }
        __syncthreads();
        for (int k = 0; k < diagonalSize; k++) // k has the same role as threadIndex???
        {
            if (max_score < H_diag_minus1[k])
            {
                max_score = H_diag_minus1[k];
                // norm_index = k = threadIndex - 1 - (diagonalIndex) + query_length
                max_pos_row = query_length - 1 - k;
                max_pos_col = diagonalIndex - max_pos_row - 1;
            }
        }
        if(DEBUG_PRINT > 10) {
            __syncthreads();
            if(threadIndex == 0)
            {
                printf("\nMax score: %d\t final_pos=%d %d\n", max_score, max_pos_col, max_pos_row);
                printf("\n-------------------------------\n");
            }
        }


    }
    __syncthreads();
    *score = max_score;
    final_pos->x = max_pos_col;
    final_pos->y = max_pos_row; 
    //-----------------------------------------------------------
    
}


namespace albp {

int antidiag(
    const std::string &seqa,
    const std::string &seqb,
    const int gap_open,
    const int gap_extend,
    const int match_score,
    const int mismatch_score,
    int* max_x = NULL, int* max_y = NULL
  )
{
    *max_x = 0;
    *max_y = 0;
    int res = 0;
    // fetch sw params
    // these are size from the matrix, including the first dummy col and row.
    const int query_length = seqa.length();
    const int target_length = seqb.length();
    const int diagonalSize = std::min(query_length + 1, target_length + 1);


    const int blockSize = 256;
    const int gridSize = (diagonalSize % blockSize) == 0 ? (diagonalSize / blockSize) : (diagonalSize / blockSize + 1);
    
    // *** memory allocation ***


    char* targetSequenceGpu;
    cudaMalloc(&targetSequenceGpu, target_length * sizeof(char));

    char* querySequenceGpu;
    cudaMalloc(&querySequenceGpu, query_length * sizeof(char));

    // result
    int3* resultsCpu;
    int3* resultsGpu;
    cudaMallocHost(&resultsCpu, sizeof(int3));
    cudaMalloc(&resultsGpu, sizeof(int3));
    cudaMemset(resultsGpu, 0, sizeof(int3));

    // compute buffers
    int* H_diag_minus2;
    int* H_diag_minus1;
    int* E_diag_minus1;
    int* F_diag_minus1;

    int number_of_threads = gridSize * blockSize + 1;

    cudaMalloc(&H_diag_minus2, number_of_threads * sizeof(int));
    cudaMalloc(&H_diag_minus1, number_of_threads * sizeof(int));
    cudaMalloc(&E_diag_minus1, number_of_threads * sizeof(int));
    cudaMalloc(&F_diag_minus1, number_of_threads * sizeof(int));

    cudaMemset(H_diag_minus2, 0, number_of_threads);
    cudaMemset(H_diag_minus1, 0, number_of_threads);
    cudaMemset(E_diag_minus1, 0, number_of_threads);
    cudaMemset(F_diag_minus1, 0, number_of_threads);

    // result
    int* score;
    int* score_gpu;
    cudaMallocHost(&score, sizeof(int));
    cudaMalloc(&score_gpu, sizeof(int));
    cudaMemset(score_gpu, 0, sizeof(int));

    int2* final_pos;
    int2* final_pos_gpu;
    cudaMallocHost(&final_pos, sizeof(int2));
    cudaMalloc(&final_pos_gpu, sizeof(int2));
    cudaMemset(final_pos_gpu, 0, sizeof(int2));


    cudaMemcpy(targetSequenceGpu, seqb.c_str(), target_length, cudaMemcpyHostToDevice);
    cudaMemcpy(querySequenceGpu, seqa.c_str(), query_length, cudaMemcpyHostToDevice);

    // --------------- compute the stuff
    cudaDeviceSynchronize();
    sw_antidiag <<<gridSize, blockSize>>>(score_gpu, final_pos_gpu, H_diag_minus2, H_diag_minus1, E_diag_minus1, F_diag_minus1,
                                         querySequenceGpu, targetSequenceGpu,
                                         query_length, target_length, match_score, mismatch_score, gap_open, gap_extend);


    // --------------- you gotta take the result back!

    cudaDeviceSynchronize();
    cudaMemcpy(score, score_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(final_pos, final_pos_gpu, sizeof(int2), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    *max_x = final_pos->y;
    *max_y = final_pos->x;
    res = *score;


    // -------------- cleanup
    cudaFree(targetSequenceGpu);
    cudaFree(querySequenceGpu);
    cudaFree(resultsGpu);
    cudaFreeHost(resultsCpu);
    
    cudaFree(H_diag_minus2);
    cudaFree(H_diag_minus1);
    cudaFree(E_diag_minus1);
    cudaFree(F_diag_minus1);


    cudaFreeHost(score);
    cudaFree(score_gpu);
    cudaFreeHost(final_pos);
    cudaFree(final_pos_gpu);

    return res;

}


}
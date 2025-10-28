#include<iostream>
#include<fstream>
#include<map>
#include<time.h>
#include<cstring>
#include <stdio.h>
#include <stdlib.h>
#include<math.h>


#define MaxIterCnt 100000
#define Epsilon 1e-100
#define inputfile "input.txt"
#define outputfile "output_GPU.txt"
#define BLOCK_SIZE 32

using namespace std;



__global__ void matrixMultiplicationGPU(const float* A, const float* B, float* C, int N , int M , int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < M; i++) {
            sum += A[row * M + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}


__host__ void CUDA_matrixmult_shared(float *source1_GPU, float* source2_GPU, float* result_GPU, int N , int M , int K)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    matrixMultiplicationGPU <<<gridSize, blockSize >>>(source1_GPU, source2_GPU, result_GPU, N , M ,K);
}

void Centered(int n, int m, float* InputMat, float* OutputMat){
    for(int i=0; i<n; i++){
        float avg = 0.0;
        for(int j=0; j<m; j++){
            avg += InputMat[i*m+j];
        }
        avg /= m;

        for(int j=0; j<m; j++){
            OutputMat[i*m+j] = InputMat[i*m+j] - avg;
        }
    }
}

void Transpose(int n, int m, float* InputMat, float* OutputMat){
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            OutputMat[j*n + i] = InputMat[i*m + j];
        }
    }
}

void Jacobi(int n, float* InputMat, float* EigenValues, float* EigenVectors, int Maxcnt, float Error){
    // Initialize Eigenvectors 
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            EigenVectors[i*n+j] = 0.0;
        }
        EigenVectors[i*n+i] = 1.0;
    }
    
    int cnt = 0;
    while(1){

        // Find Largest Element in off diagonal elements
        int p=0, q=1;
        float largest = InputMat[1];
        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                float num = fabs(InputMat[i*n+j]);
                if(i!=j && num > largest){
                    largest = num;
                    p = i;
                    q = j;
                }
            }
        }

        if(cnt > Maxcnt){
            cout << "Error: " << largest << '\n';
            break;
        }

        if(largest < Error){
            cout << "Round: " << cnt << '\n';
            cout << "Error: " << largest << '\n';
            break;
        }

        cnt++;

        // Calculate New InputMat

        float App = InputMat[p*n+p];
        float Apq = InputMat[p*n+q];
        float Aqq = InputMat[q*n+q];

        float angle = 0.5*atan2(-2*Apq, Aqq-App);
        float sinTheta = sin(angle);
        float cosTheta = cos(angle);
        float sin2Theta = sin(2*angle);
        float cos2Theta = cos(2*angle);

        InputMat[p*n+p] = App*cosTheta*cosTheta + Aqq*sinTheta*sinTheta + 2*Apq*cosTheta*sinTheta;
        InputMat[q*n+q] = App*sinTheta*sinTheta + Aqq*cosTheta*cosTheta - 2*Apq*cosTheta*sinTheta;
        InputMat[p*n+q] = 0.5*(Aqq-App)*sin2Theta + Apq*cos2Theta;
        InputMat[q*n+p] = 0.5*(Aqq-App)*sin2Theta + Apq*cos2Theta;

        for(int i=0; i<n; i++){
            if(i!=p && i!= q){
                int u = i*n+p;
                int w = i*n+q;
                float tmp = InputMat[u];
                InputMat[u] = InputMat[w]*sinTheta + tmp*cosTheta;
                InputMat[w] = InputMat[w]*cosTheta - tmp*sinTheta;
            }
        }

        for(int i=0; i<n; i++){
            if(i!=p && i!=q){
                int u = p*n + i;
                int w = q*n + i;
                float tmp = InputMat[u];
                InputMat[u] = InputMat[w]*sinTheta + tmp*cosTheta;
                InputMat[w] = InputMat[w]*cosTheta - tmp*sinTheta;
            }
        }
        // Calculate eigen vectors
        for(int i=0; i<n; i++){
            int u = i*n + p;
            int w = i*n + q;
            float tmp = EigenVectors[u];
            EigenVectors[u] = EigenVectors[w]*sinTheta + tmp*cosTheta;
            EigenVectors[w] = EigenVectors[w]*cosTheta - tmp*sinTheta;
        }

    }

    map<float, int> MP;
    for(int i=0; i<n; i++){
        EigenValues[i] = InputMat[i*n+i];

        MP.insert({EigenValues[i], i});
    }

    float* TmpVec = (float *) malloc(n * n * sizeof(float));
    map<float, int>::reverse_iterator iter = MP.rbegin();

    for(int i=0; iter != MP.rend(), i<n; iter++, i++){
        for(int j=0; j<n; j++){
            TmpVec[j*n+i] = EigenVectors[j*n + iter->second];
        }

        EigenValues[i] = iter->first;
    }

    // Set symbol
    for(int i=0; i<n; i++){
        float sum = 0.0;
        for(int j=0; j<n; j++){
            sum += TmpVec[j*n+i];
        }
        
        if(sum < 0){
            for(int j=0; j<n; j++){
                TmpVec[j*n+i] *= -1;
            }
        }
    }
    
    memcpy(EigenVectors, TmpVec, n*n*sizeof(float));
    free(TmpVec);
}

void Normalize(int n, int k, float* InputMat, float* OutputMat){
    for(int i=0; i<k; i++){
        float sum = 0.0;
        for(int j=0; j<n; j++){
            sum += InputMat[i*n+j];
        }

        for(int j=0; j<n; j++){
            OutputMat[i*n+j] = InputMat[i*n+j]/sum;
        }
    }
}


void PrintMat(int n, int m, float* Mat, ofstream &out){
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            out << Mat[i*m+j] << " ";
        }
        out << '\n';
    }
}

int main(int argc, char *argv[]) {

    ios::sync_with_stdio(false);
    cin.tie(0);

    ifstream in(inputfile);
    ofstream out(outputfile);

    int n, m, k;
    in >> n >> m >> k;
   

    struct timespec t_start, t_end;
    float elapsedTime;

    float *InputMat = (float *) malloc(n * m * sizeof(float));
    float *InputMat_Cen = (float *) malloc(n * m * sizeof(float));
    float *InputMat_Cen_T = (float *) malloc(m * n * sizeof(float));
    float *EigenVecs_Tmp = (float *) malloc(n * n * sizeof(float));
    float *EigenVecs = (float *) malloc(n * n * sizeof(float));
    float *EigenVecs_K_Norm = (float *) malloc(k * n * sizeof(float));
    float *EigenVals = (float *) malloc(n*sizeof(float));
    float *OutputMat = (float *) malloc(k * m * sizeof(float));
    float *CovMat = (float *) malloc(n * n * sizeof(float));
    
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            in >> InputMat[i*m + j];
        }
    }

    clock_gettime(CLOCK_REALTIME, &t_start);
      
    Centered(n,m,InputMat, InputMat_Cen);
    Transpose(n,m,InputMat_Cen, InputMat_Cen_T);

    /* allocate matrices in device memory and transfer matrices from host to device memory */
    float *source1_GPU, *source2_GPU, *result_GPU;
    cudaMalloc((void**)&source1_GPU, n*m*sizeof(float));
    cudaMalloc((void**)&source2_GPU, m*n*sizeof(float));
    cudaMalloc((void**)&result_GPU, n*n*sizeof(float));

    // Copy structure arrays to device
    cudaMemcpy(source1_GPU, InputMat_Cen, n*m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(source2_GPU, InputMat_Cen_T, m*n*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(result_GPU, 0, sizeof(CovMat));
    CUDA_matrixmult_shared(source1_GPU, source2_GPU, result_GPU, n , m , n); 
    cudaDeviceSynchronize();
    cudaMemcpy(CovMat, result_GPU, n*n*sizeof(float), cudaMemcpyDeviceToHost);

    Jacobi(n, CovMat, EigenVals, EigenVecs_Tmp, MaxIterCnt, Epsilon);
    Transpose(n,n,EigenVecs_Tmp, EigenVecs);
    Normalize(n,k,EigenVecs, EigenVecs_K_Norm);
    
    cudaMemset(source1_GPU, 0, sizeof(source1_GPU));
    cudaMemset(source2_GPU, 0, sizeof(source2_GPU));
    cudaMemcpy(source1_GPU, EigenVecs_K_Norm, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(source2_GPU, InputMat_Cen,   n * m *sizeof(float), cudaMemcpyHostToDevice);


    cudaMemset(result_GPU, 0, sizeof(result_GPU));
    
    CUDA_matrixmult_shared(source1_GPU, source2_GPU, result_GPU, k , n , m); 
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_REALTIME, &t_end);
    
    elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
    cout << "time: " << elapsedTime << "ms" << '\n';

    cudaMemcpy(OutputMat, result_GPU, k * m *sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    out << "Output data:\n";
    PrintMat(k,m,OutputMat, out);
    
    

    free(InputMat);
    free(InputMat_Cen);
    free(InputMat_Cen_T);
    free(EigenVecs_Tmp);
    free(EigenVecs);
    free(EigenVecs_K_Norm);
    free(EigenVals);
    free(OutputMat);
    free(CovMat);

    cudaFree(source1_GPU);
    cudaFree(source2_GPU);
    cudaFree(result_GPU);
     
    return 0;
}




#include<iostream>
#include<fstream>
#include<map>
#include<time.h>
#include<cstring>
#include<math.h>
#include<omp.h>
using namespace std;
#define MaxIterCnt 100000
#define Epsilon 1e-100
#define inputfile "input.txt"
#define outputfile "output.txt"
#define NUM_THREAD 2
void PrintMat(int n, int m, float* Mat, ofstream &out){
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            out << Mat[i*m+j] << " ";
        }
        out << '\n';
    }
}
void Transpose(int n, int m, float* InputMat, float* OutputMat){
    // #pragma omp parallel for 
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            OutputMat[j*n + i] = InputMat[i*m + j];
        }
    }
}
void MatMul(int p, int q, int r, float* A, float* B, float* C){

    
    for(int i=0; i<p; i++){
        #pragma omp parallel for num_threads(NUM_THREAD)
        for(int j=0; j<r; j++){
            float sum = 0.0;
            for(int k=0; k<q; k++){
                sum += A[i*q+k]*B[k*r+j];
            }
            C[i*r+j] = sum;
        }
    }
    
}
void Centered(int n, int m, float* InputMat, float* OutputMat){
    // #pragma omp parallel for 
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
void Jacobi(int n, float* InputMat, float* EigenValues, float* EigenVectors, int Maxcnt, float Error){
    // Initialize Eigenvectors 
    #pragma omp parallel for num_threads(NUM_THREAD)
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
        #pragma omp parallel for num_threads(NUM_THREAD) reduction(max : largest)
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

        #pragma omp parallel for num_threads(NUM_THREAD)
        for(int i=0; i<n; i++){
            if(i!=p && i!= q){
                int u = i*n+p;
                int w = i*n+q;
                float tmp = InputMat[u];
                InputMat[u] = InputMat[w]*sinTheta + tmp*cosTheta;
                InputMat[w] = InputMat[w]*cosTheta - tmp*sinTheta;
            }
        }

        #pragma omp parallel for num_threads(NUM_THREAD)
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
        #pragma omp parallel for num_threads(NUM_THREAD)
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
        #pragma omp parallel for num_threads(NUM_THREAD)
        for(int j=0; j<n; j++){
            TmpVec[j*n+i] = EigenVectors[j*n + iter->second];
        }

        EigenValues[i] = iter->first;
    }

    // Set symbol
    #pragma omp parallel for num_threads(NUM_THREAD)
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
    // #pragma omp parallel for 
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
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);

    ifstream in(inputfile);
    ofstream out(outputfile);

    int n, m, k;
    // float START, END;
    struct timespec t_start, t_end;
	float elapsedTime;
    
    in >> n >> m >> k;
    
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

    // START = clock();
    clock_gettime(CLOCK_REALTIME, &t_start);
    // Centralized
    Centered(n,m,InputMat, InputMat_Cen);
    Transpose(n,m,InputMat_Cen, InputMat_Cen_T);
    
    // Get CovMat (X*X^T)
    MatMul(n,m,n,InputMat_Cen, InputMat_Cen_T, CovMat);
    // Get Eigenvalues and Eignevectors
    Jacobi(n, CovMat, EigenVals, EigenVecs_Tmp, MaxIterCnt, Epsilon);
    Transpose(n,n,EigenVecs_Tmp, EigenVecs);

    // Normalize Eigenvectors
    Normalize(n,k,EigenVecs, EigenVecs_K_Norm);
    
    // Get the Output Matrix
    MatMul(k,n,m,EigenVecs_K_Norm, InputMat_Cen, OutputMat);

    // END = clock();

    // Print result
    // cout << "Time: " << (END-START)/CLOCKS_PER_SEC << '\n';
    clock_gettime(CLOCK_REALTIME, &t_end);
    elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;

    cout << "time: " << elapsedTime << "ms" << '\n';
    
    
    free(InputMat);
    free(InputMat_Cen);
    free(InputMat_Cen_T);
    free(EigenVecs_Tmp);
    free(EigenVecs);
    free(EigenVecs_K_Norm);
    free(EigenVals);
    free(OutputMat);
    free(CovMat);

}
#include<iostream>
#include<fstream>
#include<map>
#include<time.h>
#include<cstring>
#include<math.h>
#include<pthread.h>

using namespace std;
#define MaxIterCnt 100000
#define Epsilon 1e-100
#define inputfile "input.txt"
#define outputfile "output.txt"
#define NUM_THREADS 4
float *InputMat_Cen, *InputMat_Cen_T, *CovMat, *EigenVecs_K_Norm, *OutputMat,*EigenVecs_Tmp,*EigenVals;
float* maxval = (float *)malloc(NUM_THREADS * sizeof(int));
int* mp = (int *)malloc(NUM_THREADS * sizeof(int));
int* mq = (int *)malloc(NUM_THREADS * sizeof(int));

struct v{
    int start;
    int end; 
    int r; 
    int q;
};
struct u{
    int start;
    int end; 
    int n;
    int id;
};
void PrintMat(int n, int m, float* Mat, ofstream &out){
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            out << Mat[i*m+j] << " ";
        }
        out << '\n';
    }
}
void Transpose(int n, int m, float* InputMat, float* OutputMat){
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            OutputMat[j*n + i] = InputMat[i*m + j];
        }
    }
}
void *runner(void *param){
    struct v* data = (struct v*)param;
    for(int i=data->start; i<data->end; i++){
        for(int j=0; j<data->r; j++){
            float sum = 0.0;
            for(int k=0; k<data->q; k++){
                sum += InputMat_Cen[i*data->q+k]*InputMat_Cen_T[k*data->r+j];
            }    
            CovMat[i*data->r+j] = sum;
        }
    }
}
void MatMul_pthread(int p, int q, int r){

    pthread_t tid[NUM_THREADS];
    pthread_attr_t attr[NUM_THREADS];
    int chunk = p/NUM_THREADS;
    for(int i=0; i<NUM_THREADS; i++){
        struct v* data = (struct v*) malloc(sizeof(struct v));
        data->start = i*(chunk);
        data->end = (i+1)*chunk;
        data->r = r;
        data->q = q;
        if(data->end > p){
            data->end = p;
        }
        pthread_attr_init(&attr[i]);
        pthread_create(&tid[i], &attr[i], runner, data);
    }

    for(int i=0; i<NUM_THREADS; i++){
        pthread_join(tid[i], NULL);
    }
}

void *runner1(void *param){

    struct v* data = (struct v*)param;
    for(int i=data->start; i<data->end; i++){
        for(int j=0; j<data->r; j++){
            float sum = 0.0;
            for(int k=0; k<data->q; k++){
                sum += EigenVecs_K_Norm[i*data->q+k]*InputMat_Cen[k*data->r+j];
            }    
            OutputMat[i*data->r+j] = sum;
        }
    }
}
void *getmax(void *param){

    struct u* data = (struct u*)param;
    for(int i=data->start; i<data->end; i++){
        for(int j=0; j<data->n; j++){
            float num = fabs(CovMat[i*data->n+j]);
            if(i!=j && num > maxval[data->id]){
                maxval[data->id] = num;
                mp[data->id] = i;
                mq[data->id] = j;
            }
        }
    }
}
void MatMul_pthread1(int p, int q, int r){

    pthread_t tid[NUM_THREADS];
    pthread_attr_t attr[NUM_THREADS];
    int chunk = p/NUM_THREADS;
    for(int i=0; i<NUM_THREADS; i++){
        struct v* data = (struct v*) malloc(sizeof(struct v));
        data->start = i*(chunk);
        data->end = (i+1)*chunk;
        data->r = r;
        data->q = q;
        if(data->end > p){
            data->end = p;
        }
        pthread_attr_init(&attr[i]);
        pthread_create(&tid[i], &attr[i], runner1, data);
    }

    for(int i=0; i<NUM_THREADS; i++){
        pthread_join(tid[i], NULL);
    }
}
void MatMul(int p, int q, int r, float* A, float* B, float* C){
    for(int i=0; i<p; i++){
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
void Jacobi(int n, int Maxcnt, float Error){
    // Initialize Eigenvectors 
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            EigenVecs_Tmp[i*n+j] = 0.0;
        }
        EigenVecs_Tmp[i*n+i] = 1.0;
    }
    
    int cnt = 0;
    while(1){

        // Find Largest Element in off diagonal elements
        int p=0,q=0;
        int largest=0;
        for(int i=0; i<NUM_THREADS; i++){
            maxval[i] = 0;
            mp[i] = 0;
            mq[i] = 0;
        }
        pthread_t tid[NUM_THREADS];
        pthread_attr_t attr[NUM_THREADS];
        int chunk = n/NUM_THREADS;
        for(int i=0; i<NUM_THREADS; i++){
            struct u* data = (struct u*) malloc(sizeof(struct u));
            data->start = i*(chunk);
            data->end = (i+1)*chunk;
            data->id = i;
            data->n = n;
            if(data->end > n){
                data->end = n;
            }
            pthread_attr_init(&attr[i]);
            pthread_create(&tid[i], &attr[i], getmax, data);
        }

        for(int i=0; i<NUM_THREADS; i++){
            pthread_join(tid[i], NULL);
        }


        for(int i=0; i<NUM_THREADS; i++){
            if(maxval[i] > largest){
                largest = maxval[i];
                p = mp[i];
                q = mq[i];
            }
        }
        // cout << p << " " << q;
        

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

        float App = CovMat[p*n+p];
        float Apq = CovMat[p*n+q];
        float Aqq = CovMat[q*n+q];

        float angle = 0.5*atan2(-2*Apq, Aqq-App);
        float sinTheta = sin(angle);
        float cosTheta = cos(angle);
        float sin2Theta = sin(2*angle);
        float cos2Theta = cos(2*angle);

        CovMat[p*n+p] = App*cosTheta*cosTheta + Aqq*sinTheta*sinTheta + 2*Apq*cosTheta*sinTheta;
        CovMat[q*n+q] = App*sinTheta*sinTheta + Aqq*cosTheta*cosTheta - 2*Apq*cosTheta*sinTheta;
        CovMat[p*n+q] = 0.5*(Aqq-App)*sin2Theta + Apq*cos2Theta;
        CovMat[q*n+p] = 0.5*(Aqq-App)*sin2Theta + Apq*cos2Theta;

        for(int i=0; i<n; i++){
            if(i!=p && i!= q){
                int u = i*n+p;
                int w = i*n+q;
                float tmp = CovMat[u];
                CovMat[u] = CovMat[w]*sinTheta + tmp*cosTheta;
                CovMat[w] = CovMat[w]*cosTheta - tmp*sinTheta;
            }
        }

        for(int i=0; i<n; i++){
            if(i!=p && i!=q){
                int u = p*n + i;
                int w = q*n + i;
                float tmp = CovMat[u];
                CovMat[u] = CovMat[w]*sinTheta + tmp*cosTheta;
                CovMat[w] = CovMat[w]*cosTheta - tmp*sinTheta;
            }
        }


        // Calculate eigen vectors
        for(int i=0; i<n; i++){
            int u = i*n + p;
            int w = i*n + q;
            float tmp = EigenVecs_Tmp[u];
            EigenVecs_Tmp[u] = EigenVecs_Tmp[w]*sinTheta + tmp*cosTheta;
            EigenVecs_Tmp[w] = EigenVecs_Tmp[w]*cosTheta - tmp*sinTheta;
        }

    }

    map<float, int> MP;
    for(int i=0; i<n; i++){
        EigenVals[i] = CovMat[i*n+i];

        MP.insert({EigenVals[i], i});
    }

    float* TmpVec = (float *) malloc(n * n * sizeof(float));
    map<float, int>::reverse_iterator iter = MP.rbegin();

    for(int i=0; iter != MP.rend(), i<n; iter++, i++){
        for(int j=0; j<n; j++){
            TmpVec[j*n+i] = EigenVecs_Tmp[j*n + iter->second];
        }

        EigenVals[i] = iter->first;
    }

    // Set symbol
    // for(int i=0; i<n; i++){
    //     float sum = 0.0;
    //     for(int j=0; j<n; j++){
    //         sum += TmpVec[j*n+i];
    //     }
        
    //     if(sum < 0){
    //         for(int j=0; j<n; j++){
    //             TmpVec[j*n+i] *= -1;
    //         }
    //     }
    // }
    
    memcpy(EigenVecs_Tmp, TmpVec, n*n*sizeof(float));
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
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);

    ifstream in(inputfile);
    ofstream out(outputfile);

    int n, m, k;
    struct timespec t_start, t_end;
	float elapsedTime;

    
    in >> n >> m >> k;
    
    float *InputMat = (float *) malloc(n * m * sizeof(float));
    InputMat_Cen = (float *) malloc(n * m * sizeof(float));
    InputMat_Cen_T = (float *) malloc(m * n * sizeof(float));
    EigenVecs_Tmp = (float *) malloc(n * n * sizeof(float));
    float *EigenVecs = (float *) malloc(n * n * sizeof(float));
    EigenVecs_K_Norm = (float *) malloc(k * n * sizeof(float));
    EigenVals = (float *) malloc(n*sizeof(float));
    OutputMat = (float *) malloc(k * m * sizeof(float));
    CovMat = (float *) malloc(n * n * sizeof(float));
    
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            in >> InputMat[i*m + j];
        }
    }

    // clock_gettime(CLOCK_REALTIME, &t_start);
    clock_gettime(CLOCK_REALTIME, &t_start);
    // Centralized
    Centered(n,m,InputMat, InputMat_Cen);
    Transpose(n,m,InputMat_Cen, InputMat_Cen_T);
    
    // Get CovMat (X*X^T)
    MatMul_pthread(n,m,n);
    // Get Eigenvalues and Eignevectors
    Jacobi(n, MaxIterCnt, Epsilon);
    Transpose(n,n,EigenVecs_Tmp, EigenVecs);

    // Normalize Eigenvectors
    Normalize(n,k,EigenVecs, EigenVecs_K_Norm);
    
    // Get the Output Matrix
    MatMul_pthread1(k,n,m);
    clock_gettime(CLOCK_REALTIME, &t_end);
    
    
    elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;

    
    cout << "time: " << elapsedTime << "ms" << '\n';

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

}
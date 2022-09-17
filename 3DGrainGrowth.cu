#include<iostream>
#include<cstdlib>
#include<math.h>
#include<string>
#include<iostream>
#include<fstream>
#include<stdio.h>
// check function
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

// specify parameters
int Nx = 64, Ny = 64, Nz = 64; // system dimension
int Norient = 36; // number of orientations
// coefficient
float h_alpha = 1, h_beta = 1, h_gamma = 1; // alpha, beta, gamma
float kxx = 2, kyy = 2, kzz = 2; //gradient coefficient
// simulation settings
float dx = 2, dt = 0.1;
int Nstep = 200, Noutput = 20;

// specify device parameters
__constant__ int d_Nx, d_Ny, d_Nz;
__constant__ int d_Norient;
__constant__ float d_kxx, d_kyy, d_kzz;
__constant__ float d_alpha, d_beta, d_gamma;
__constant__ float d_dx,d_dt;

// CPU functions
//4D index to 1D index
__host__ int hconvert4Dindex(int i, int j, int k, int n) {
    int index1D;
    index1D = i * Ny * Nz * Norient + j * Nz * Norient + k * Norient + n;
    return index1D;
}
__device__ int dconvert4Dindex(int i, int j, int k, int n) {
    int index1D;
    index1D = i * d_Ny * d_Nz * d_Norient + j * d_Nz * d_Norient + k * d_Norient + n;
    return index1D;
}
//3D index to 1D index
__host__ int convert3Dindex(int i, int j, int k) {
    int index1D;
    index1D = i * Ny * Nz + j * Nz + k;
    return index1D;
}

// initialization of eta
__host__ void initialize(float *eta) {
    srand(30);
    for (int i = 0; i < Nx * Ny * Nz * Norient; i++) {
        eta[i] = ((float)rand() /RAND_MAX) * 0.002 - 0.001; // eta in the range of (-0.001, 0.001)
    }
}

// get poshi (indicator of grain or grain boundary) based on eqn. (6)
__host__ float outputgrainvol(float *eta, float *grainvol, int step) {
    int indexeta, indexgrainvol;
    float avegrainvol = 0;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                // get corre
                indexgrainvol = convert3Dindex(i, j, k);
                grainvol[indexgrainvol] = 0;
                for (int n = 0; n < Norient; n++) {
                    indexeta = hconvert4Dindex(i, j, k, n);
                    grainvol[indexgrainvol] += pow(eta[indexeta], 2);
                }
                // calcualate the average grainvol
                avegrainvol += grainvol[indexgrainvol];
            }
        }
    }
    avegrainvol = avegrainvol / (Nx * Ny * Nz);
    // output the grainvol
    std::string filename = "grainvol" + std::to_string(step) + ".txt";
    std::ofstream outputfile;
    outputfile.open(filename);
    outputfile << "x" << " " << "y" << " " << "z" << " " << "grainvol" << std::endl;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                indexgrainvol = convert3Dindex(i, j, k);
                outputfile << i << " " << j << " " << k << " " << grainvol[indexgrainvol] << std::endl;
            }
        }
    }
    outputfile.close();

    return avegrainvol;
    
}

__global__
void calRHS(float *eta, float *RHS){
    // get current index
    int index;
    index = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("%d %d %d %d \n", index, blockIdx.x, blockDim.x, threadIdx.x);
    // do calculation if the index is in the range
    if (index < d_Nx * d_Ny * d_Nz * d_Norient){
        // calcualte the i, j, k, n index first
        int index_temp;
        index_temp = index;
        int i0, j0, k0, n0;
        n0 = index_temp % d_Norient;
        index_temp = index_temp / d_Norient;
        k0 = index_temp % d_Nz;
        index_temp = index_temp / d_Nz;
        j0 = index_temp % d_Ny;
        index_temp = index_temp / d_Ny;
        i0 = index_temp;
        // calculate the volume energy
        int index_othereta;
        // alpha and beta terms
        RHS[index] = d_alpha * eta[index] - d_beta * pow(eta[index], 3);
        //printf("%d %d %d %d %d\n", index, i0, j0, k0, n0);
	// gamma terms
        for (int ntemp = 0; ntemp < d_Norient; ntemp++) {
            if (ntemp != n0) {
                index_othereta = dconvert4Dindex(i0, j0, k0, ntemp);
                RHS[index] = RHS[index] -2 * d_gamma * eta[index] * pow(eta[index_othereta], 2);
		//printf("%.8f\n", d_gamma * eta[index]*eta[index_othereta]*eta[index_othereta]);
            }
        }
	//printf("%d %.8f\n", index, RHS[index]);
        // calculate the gradient energy
        int index_x1, index_x2;
        int index_y1, index_y2;
        int index_z1, index_z2;
        bool x1D, y1D, z1D;
        // check the dimension
        if (d_Nx != 1) {x1D = false;} else {x1D = true;}
        if (d_Ny != 1) {y1D = false;} else {y1D = true;}
        if (d_Nz != 1) {z1D = false;} else {z1D = true;}
        // get index of neighbors first
        // Implemenet periodic boundary condition
        if (x1D == false) {
            index_x1 = i0 - 1;
            if (index_x1 < 0) {index_x1 = index_x1 + d_Nx;}
            index_x2 = i0 + 1;
            if (index_x2 >= d_Nx) {index_x2 = index_x2 - d_Nx;}
        }
        if (y1D == false) {
            index_y1 = j0 - 1;
            if (index_y1 < 0) {index_y1 = index_y1 + d_Ny;}
            index_y2 = j0 + 1;
            if (index_y2 >= d_Ny) {index_y2 = index_y2 - d_Ny;}
        }
        if (z1D == false) {
            index_z1 = k0 - 1;
            if (index_z1 < 0) {index_z1 = index_z1 + d_Nz;}
            index_z2 = k0 + 1;
            if (index_z2 >= d_Nz) {index_z2 = index_z2 - d_Nz;}
        }
        // calculate laplace of eta
        int indextemp1, indextemp2;
        if (x1D == false) {
            indextemp1 = dconvert4Dindex(index_x1, j0, k0, n0);
            indextemp2 = dconvert4Dindex(index_x2, j0, k0, n0);
            RHS[index] = RHS[index] + d_kxx * (eta[indextemp1] + eta[indextemp2] - 2 * eta[index]) / pow(d_dx, 2);
        }
	//printf("%d %.8f\n", index, RHS[index]);
        if (y1D == false) {
            indextemp1 = dconvert4Dindex(i0, index_y1, k0, n0);
            indextemp2 = dconvert4Dindex(i0, index_y2, k0, n0);
            RHS[index] = RHS[index] + d_kyy * (eta[indextemp1] + eta[indextemp2] - 2 * eta[index]) / pow(d_dx, 2);
        }
	//printf("%d %.8f\n", index, RHS[index]);
        if (z1D == false) {
            indextemp1 = dconvert4Dindex(i0, j0, index_z1, n0);
            indextemp2 = dconvert4Dindex(i0, j0, index_z2, n0);
            RHS[index] = RHS[index] + d_kzz * (eta[indextemp1] + eta[indextemp2] - 2 * eta[index]) / pow(d_dx, 2);
        }
	//printf("%d %.8f\n", index, RHS[index]);
        // update eta
	//printf("%d %.8f %.8f\n", index, eta[index], RHS[index]);
        //eta[index] = eta[index] + d_dt * RHS[index];
	//eta[index] = 1;
	//printf("%d %.8f %.8f\n", index, eta[index], RHS[index]);
    }
}

__global__
void updateeta(float *eta, float *RHS){
    // get current index
    int index;
    index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < d_Nx * d_Ny * d_Nz * d_Norient){
	  eta[index] = eta[index] + d_dt * RHS[index];
    }
}


int main(){
    // specify host arrays
    float *eta, *RHS, *grainvol;
    eta = (float*)malloc(Nx*Ny*Nz*Norient*sizeof(float));
    RHS = (float*)malloc(Nx*Ny*Nz*Norient*sizeof(float));
    grainvol = (float*)malloc(Nx*Ny*Nz*sizeof(float));
    // specify device arrays
    float *d_eta, *d_RHS;
    checkCuda(cudaMalloc(&d_eta, Nx*Ny*Nz*Norient*sizeof(float)));
    checkCuda(cudaMalloc(&d_RHS, Nx*Ny*Nz*Norient*sizeof(float)));
    //copy parameters
    checkCuda(cudaMemcpyToSymbol(d_Nx, &Nx, sizeof(int), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_Ny, &Ny, sizeof(int), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_Nz, &Nz, sizeof(int), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_Norient, &Norient, sizeof(int), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_kxx, &kxx, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_kyy, &kyy, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_kzz, &kzz, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_alpha, &h_alpha, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_beta, &h_beta, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_gamma, &h_gamma, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_dx, &dx, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_dt, &dt, sizeof(float), 0, cudaMemcpyHostToDevice));
    // average grainvol
    float avegrainvol;
    // initialize
    initialize(eta);
    // initialize cuda time
    float milliseconds;
    cudaEvent_t startEvent, stopEvent;
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));
    // loop
    checkCuda(cudaEventRecord(startEvent,0));
    // copy variables
    checkCuda(cudaMemcpy(d_eta, eta, Nx*Ny*Nz*Norient*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_RHS, RHS, Nx*Ny*Nz*Norient*sizeof(float), cudaMemcpyHostToDevice));
    // output
    outputgrainvol(eta, grainvol,0);
    std::ofstream grainfile;
    grainfile.open("average.txt");
    grainfile << "#step" << " " << "average grain vol" << std::endl;
    // specify dimension
    int blocksize = 256;
    int numblocks = (Nx * Ny * Nz * Norient +blocksize-1)/ blocksize;
    std::cout<<blocksize<<" "<<numblocks<<std::endl;
    //float milliseconds;
    //cudaEvent_t startEvent, stopEvent;
    //checkCuda(cudaEventCreate(&startEvent));
    //checkCuda(cudaEventCreate(&stopEvent));
    // loop
    checkCuda(cudaEventRecord(startEvent,0));
    for (int s = 1; s <= Nstep; s++){
	//checkCuda(cudaEventRecord(startEvent, 0));
        calRHS<<<numblocks, blocksize>>>(d_eta, d_RHS);
	updateeta<<<numblocks, blocksize>>>(d_eta, d_RHS);
	//checkCuda(cudaEventRecord(stopEvent, 0));
	//checkCuda(cudaEventSynchronize(stopEvent));
	//checkCuda(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
	//printf("Average time(ms):%f\n", milliseconds);
        if (s % Noutput == 0){
            checkCuda(cudaMemcpy(eta, d_eta, Nx*Ny*Nz*Norient*sizeof(float), cudaMemcpyDeviceToHost));
	    //for (int te = 0; te < Norient; te++){
	    //	    printf("%.16f\n", eta[te]);
            //}
	    avegrainvol = outputgrainvol(eta, grainvol, s);
	    //printf("%.16f\n", grainvol[0]);
            grainfile << s << " " << avegrainvol << std::endl;
            std::cout << s << " " << avegrainvol << std::endl;
        }
    }
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
    
    grainfile <<"Total Loop time " << milliseconds << "ms" << std::endl;
    std::cout <<"Total Loop time " << milliseconds << "ms" << std::endl;

    free(eta);
    free(RHS);
    free(grainvol);
    cudaFree(d_eta);
    cudaFree(d_RHS);

    return 0;
}




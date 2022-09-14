// GrainGrowth.cpp : This file contains the 'main' function. Program execution begins and ends there.
// The code is built based on paper 
// Krill Iii, C. E., and L-Q. Chen. "Computer simulation of 3-D grain growth using a phase-field model." Acta materialia 50.12 (2002): 3059-3075.
// Author: Minyi Dai 
// Date: 09/04/22
//

#include<iostream>
#include<cstdlib>
#include<math.h>
#include<string>
#include<iostream>
#include<fstream>
#include<chrono>
// specify parameters
// system
int Nx = 64, Ny = 64, Nz = 1; // system dimension
int Norient = 36; // number of orientations
// coefficient
double alpha = 1, beta = 1, gamma = 1; // alpha, beta, gamma
double kxx = 2, kyy = 2, kzz = 2; //gradient coefficient
// simulation settings
double dx = 2, dt = 0.1;
int Nstep = 400, Noutput = 20;

//functions
//4D index to 1D index
int convert4Dindex(int i, int j, int k, int n) {
    int index1D;
    index1D = i * Ny * Nz * Norient + j * Nz * Norient + k * Norient + n;
    return index1D;
}
//3D index to 1D index
int convert3Dindex(int i, int j, int k) {
    int index1D;
    index1D = i * Ny * Nz + j * Nz + k;
    return index1D;
}

// initialization of eta
void initialize(double* eta) {
    srand(30);
    for (int i = 0; i < Nx * Ny * Nz * Norient; i++) {
        eta[i] = ((double)rand() /RAND_MAX) * 0.002 - 0.001; // eta in the range of (-0.001, 0.001)
    }
}

// calculate the field due to volume energy (based on eqn. (3))
void calvolfield(double* eta, double* volfield) {
    int index1D, index1Dtemp;
    // alpha and beta terms
    for (int i = 0; i < Nx * Ny * Nz * Norient; i++) {
        volfield[i] = alpha * eta[i] - beta * pow(eta[i], 3);
    }
    // gamma terms
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                for (int n = 0; n < Norient; n++) {
                    index1D = convert4Dindex(i, j, k, n);
                    for (int ntemp = 0; ntemp < Norient; ntemp++) {
                        if (ntemp != n) {
                            index1Dtemp = convert4Dindex(i, j, k, ntemp);
                            volfield[index1D] += -2 * gamma * eta[index1D] * pow(eta[index1Dtemp], 2);
                        }
                    }
                }
            }
        }
    }
}

// calculate the field due to gradient energy (based on eqn. (3))
void calgrafield(double* eta, double* gradfield) {
    int index1D, index1Dtemp1, index1Dtemp2;
    int index_x1, index_x2;
    int index_y1, index_y2;
    int index_z1, index_z2;
    bool x1D, y1D, z1D;
    // check the dimension
    if (Nx != 1) {
        x1D = false;
    }
    else {
        x1D = true;
    }
    if (Ny != 1) {
        y1D = false;
    }
    else {
        y1D = true;
    }
    if (Nz != 1) {
        z1D = false;
    }
    else {
        z1D = true;
    }

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                // get index of neighbors first
                // Implemenet periodic boundary condition
                if (x1D == false) {
                    index_x1 = i - 1;
                    if (index_x1 < 0) {
                        index_x1 = index_x1 + Nx;
                    }
                    index_x2 = i + 1;
                    if (index_x2 >= Nx) {
                        index_x2 = index_x2 - Nx;
                    }
                }
                if (y1D == false) {
                    index_y1 = j - 1;
                    if (index_y1 < 0) {
                        index_y1 = index_y1 + Ny;
                    }
                    index_y2 = j + 1;
                    if (index_y2 >= Ny) {
                        index_y2 = index_y2 - Ny;
                    }
                }
                if (z1D == false) {
                    index_z1 = k - 1;
                    if (index_z1 < 0) {
                        index_z1 = index_z1 + Nz;
                    }
                    index_z2 = k + 1;
                    if (index_z2 >= Nz) {
                        index_z2 = index_z2 - Nz;
                    }
                }
                // calculate laplace of eta
                for (int n = 0; n < Norient; n++) {
                    index1D = convert4Dindex(i, j, k, n);
                    // set the value to be zero first.
                    gradfield[index1D] = 0;
                    if (x1D == false) {
                        index1Dtemp1 = convert4Dindex(index_x1, j, k, n);
                        index1Dtemp2 = convert4Dindex(index_x2, j, k, n);
                        gradfield[index1D] += (eta[index1Dtemp1] + eta[index1Dtemp2] - 2 * eta[index1D]) / pow(dx, 2);
                    }
                    if (y1D == false) {
                        index1Dtemp1 = convert4Dindex(i, index_y1, k, n);
                        index1Dtemp2 = convert4Dindex(i, index_y2, k, n);
                        gradfield[index1D] += (eta[index1Dtemp1] + eta[index1Dtemp2] - 2 * eta[index1D]) / pow(dx, 2);
                    }
                    if (z1D == false) {
                        index1Dtemp1 = convert4Dindex(i, j, index_z1, n);
                        index1Dtemp2 = convert4Dindex(i, j, index_z2, n);
                        gradfield[index1D] += (eta[index1Dtemp1] + eta[index1Dtemp2] - 2 * eta[index1D]) / pow(dx, 2);
                    }
                }
            }
        }
    }
}


// update eta
void updateeta(double* eta, double* volfield, double* gradfield) {
    //time-marching algorithm: forward Euler
    for (int i = 0; i < Nx * Ny * Nz * Norient; i++) {
        eta[i] = eta[i] + dt * (volfield[i] + gradfield[i]);
    }
}

// get poshi (indicator of grain or grain boundary) based on eqn. (6)
double outputgrainvol(double* eta, double* grainvol, int step) {
    int indexeta, indexgrainvol;
    double avegrainvol = 0;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                // get corre
                indexgrainvol = convert3Dindex(i, j, k);
                grainvol[indexgrainvol] = 0;
                for (int n = 0; n < Norient; n++) {
                    indexeta = convert4Dindex(i, j, k, n);
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


int main()
{
    auto t_start = std::chrono::high_resolution_clock::now();
    // specify arrays
    double *eta = new double[Nx * Ny * Nz * Norient]; // eta
    double *grainvol = new double[Nx * Ny * Nz]; // posi in eqn. (6)
    double *volfield = new double[Nx * Ny * Nz * Norient]; // field due to volume energy
    double *gradfield = new double[Nx * Ny * Nz * Norient]; // field due to gradient energy
    // average grainvol
    double avegrainvol;
    // intialize
    initialize(eta);
    outputgrainvol(eta, grainvol, 0);
    // loop
    // open another file to document avegrainvol
    std::ofstream grainfile;
    grainfile.open("average.txt");
    grainfile << "#step" << " " << "average grain vol" << std::endl;
    for (int s = 1; s <= Nstep; s++) {
        // calculate RHS related to the volume energy
        calvolfield(eta, volfield);
        // calculate RHS related to gradient energy
        calgrafield(eta, gradfield);
        // update eta
        updateeta(eta, volfield, gradfield);
        // output gain volume
        if (s % Noutput == 0) {
            avegrainvol = outputgrainvol(eta, grainvol, s);
            grainfile << s << " " << avegrainvol << std::endl;
            std::cout << s << " " << avegrainvol << std::endl;
        }
    }
    // delete
    delete[] eta;
    delete[] grainvol;
    delete[] volfield;
    delete[] gradfield;
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << Nstep << " steps:  " << elapsed_time_ms << " ms" << std::endl;
    grainfile << Nstep << " steps:  " << elapsed_time_ms << " ms" << std::endl;
    grainfile.close();
    return 0;
}

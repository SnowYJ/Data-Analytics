#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "NBody.h"
#include "NBodyVisualiser.h"
#include "cuda_texture_types.h"
#include "texture_fetch_functions.hpp"

#define USER_NAME "acz19yz"
#define BLOCK_SIZE 32

void print_help();

// command checker.
void command_check_NGM(int argc, char* argv[]);
void command_check_IF(int argc, char* argv[]);
void body_initializer(char* argv[]);

// strsep
char* strsep(char** stringp, const char* delim);
// error
void error_func();
// is integer
boolean is_int(char* str);

// serial
void serial_step(void);
void openMP_step(void);
void cuda_step(void);

void pointer_operation_start();
void pointer_operation_end();

// N, D, M and I.
int num; 
int grid; 
MODE mode; 
int iter = 1; 

// index of I or file_name.
int index_arg_f = 0;

// running visualizer if true.
boolean toggle = FALSE;
// command has argument -i or -f.
boolean cond_arg_i = FALSE;
boolean cond_arg_f = FALSE;

// host pointer.
nbody *body_pointer; 
nbody *temp_pointer;
float *acti_map;

// device variable.
__device__ __constant__ int nbody_count;
__device__ __constant__ int nbody_grid;

// device pointer.
nbody* d_body_pointer;
nbody* d_temp_pointer;

nbody_soa d_soa_body_pointer;
nbody_soa d_soa_temp_pointer;

float* d_acti_map;

// using texture memory.
texture<float, cudaTextureType1D, cudaReadModeElementType> arr_x;
texture<float, cudaTextureType1D, cudaReadModeElementType> arr_y;
texture<float, cudaTextureType1D, cudaReadModeElementType> arr_vx;
texture<float, cudaTextureType1D, cudaReadModeElementType> arr_vy;
texture<float, cudaTextureType1D, cudaReadModeElementType> arr_m;

// convert AoS to SoA for better bandwidth.
__global__ void AoS_to_SoA(nbody_soa d_soa_body_pointer, nbody* d_body_pointer) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	d_soa_body_pointer.x[i] = d_body_pointer[i].x;
	d_soa_body_pointer.y[i] = d_body_pointer[i].y;
	d_soa_body_pointer.vx[i] = d_body_pointer[i].vx;
	d_soa_body_pointer.vy[i] = d_body_pointer[i].vy;
	d_soa_body_pointer.m[i] = d_body_pointer[i].m;
}

// density array should be initialized every interation.
__global__ void cuda_reset_density(float* d_acti_map) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	d_acti_map[i] = 0.0f;
}

// force calculation function.
__global__ void cuda_step_texture_SoA(nbody_soa d_soa_temp_pointer, float* d_acti_map) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float unit = 1.0f / nbody_grid;

	float body_x = tex1Dfetch(arr_x, i);
	float body_y = tex1Dfetch(arr_y, i);
	float body_vx = tex1Dfetch(arr_vx, i);
	float body_vy = tex1Dfetch(arr_vy, i);
	float body_m = tex1Dfetch(arr_m, i);

	float force_x = 0.0f;
	float force_y = 0.0f;

	// read from texture memory.
	for (int j = 0; j < nbody_count; j++) {
		if (j == i) continue;
		float buffer_x = tex1Dfetch(arr_x, j);
		float buffer_y = tex1Dfetch(arr_y, j);
		float buffer_m = tex1Dfetch(arr_m, j);
		float dis_x = buffer_x - body_x;
		float dis_y = buffer_y - body_y;
		float magnitude = (float)sqrt((double)dis_x * dis_x + (double)dis_y * dis_y);
		force_x += (buffer_m * dis_x) / (float)pow(((double)magnitude + (double)SOFTENING*SOFTENING), 3.0 / 2);
		force_y += (buffer_m * dis_y) / (float)pow(((double)magnitude + (double)SOFTENING*SOFTENING), 3.0 / 2);
	}
	force_x *= G * body_m;
	force_y *= G * body_m;

	float acc_x = force_x / body_m;
	float acc_y = force_y / body_m;

	float new_vx = body_vx + dt * acc_x;
	float new_vy = body_vy + dt * acc_y;
	float new_x = body_x + dt * new_vx;
	float new_y = body_y + dt * new_vy;

	// write to d_soa_temp_pointer in global memory.
	d_soa_temp_pointer.vx[i] = new_vx;
	d_soa_temp_pointer.vy[i] = new_vy;
	d_soa_temp_pointer.x[i] = new_x;
	d_soa_temp_pointer.y[i] = new_y;

	// update density.
	if (new_x > 0 && new_x < 1 && new_y > 0 && new_y < 1) {
		int index_y = (int) floor(new_y / unit);
		int index_x = (int) floor(new_x / unit);
		int index = nbody_grid * index_y + index_x;
		atomicAdd(&d_acti_map[index], (float) nbody_grid / nbody_count);
	}
}

void swap_pointer() {
	float* temp_x, * temp_y, * temp_vx, * temp_vy;

	temp_x = d_soa_temp_pointer.x;
	d_soa_temp_pointer.x = d_soa_body_pointer.x;
	d_soa_body_pointer.x = temp_x;

	temp_y = d_soa_temp_pointer.y;
	d_soa_temp_pointer.y = d_soa_body_pointer.y;
	d_soa_body_pointer.y = temp_y;

	temp_vx = d_soa_temp_pointer.vx;
	d_soa_temp_pointer.vx = d_soa_body_pointer.vx;
	d_soa_body_pointer.vx = temp_vx;

	temp_vy = d_soa_temp_pointer.vy;
	d_soa_temp_pointer.vy = d_soa_body_pointer.vy;
	d_soa_body_pointer.vy = temp_vy;
}

void cuda_step(void) {
	int num_block_simualtion = (int) ceil(num/(double)BLOCK_SIZE);
	int num_block_density = (int) ceil((grid*grid)/(double)BLOCK_SIZE);

	dim3 threadsPerBlock(BLOCK_SIZE, 1, 1);
	dim3 blocksPerGrid(num_block_simualtion, 1, 1);
	dim3 blocksPerGrid1(num_block_density, 1, 1);

	for (int i = 0; i < iter; i++) {
		// reset density.
		cuda_reset_density << <blocksPerGrid1, threadsPerBlock >> > (d_acti_map);
		cudaDeviceSynchronize();
		// update temp nbody.
		cuda_step_texture_SoA << <blocksPerGrid, threadsPerBlock >> > (d_soa_temp_pointer, d_acti_map);
		cudaDeviceSynchronize();
		// update d_body_pointer.
		swap_pointer();
	}
}

int main(int argc, char *argv[]) {
	
	// function: check command n, g, m.
	command_check_NGM(argc, argv);

	//  n, g, m initialization & allocate heap
	num = atoi(argv[1]);
	grid = atoi(argv[2]);
	if (strcmp(argv[3], "CPU") == 0) mode = CPU; 
	if (strcmp(argv[3], "OPENMP") == 0) mode = OPENMP; 
	if (strcmp(argv[3], "CUDA") == 0) mode = CUDA; 

	body_pointer = (struct nbody *) malloc(sizeof(struct nbody) * num);
	// used in serial and openmp step function.
	temp_pointer = (struct nbody*) malloc(sizeof(struct nbody) * num);
	acti_map = (float *)malloc(sizeof(float) * (grid * grid));

	// function: check command -i -f. 
	command_check_IF(argc,argv);

	// function: read file or random initialization 
	body_initializer(argv);

	// function: data move to device and texture memory bind.
	pointer_operation_start();

	// convert AoS to SoA.
	dim3 threadsPerBlock(BLOCK_SIZE, 1, 1);
	dim3 blocksPerGrid((int)ceil(num/(double)BLOCK_SIZE), 1, 1);
	AoS_to_SoA << < blocksPerGrid, threadsPerBlock >> > (d_soa_body_pointer, d_body_pointer);

	// running visualizer or iteration simulation
	if (toggle) {
		if (mode == 0) {
			initViewer(num, grid, mode, &serial_step);
			setNBodyPositions(body_pointer);
			setHistogramData(acti_map);
			startVisualisationLoop();
		}
		else if (mode == 1) {
			initViewer(num, grid, mode, &openMP_step);
			setNBodyPositions(body_pointer);
			setHistogramData(acti_map);
			startVisualisationLoop();
		}
		else {
			initViewer(num, grid, mode, &cuda_step);
			setNBodyPositions2f(d_soa_body_pointer.x, d_soa_body_pointer.y);
			setActivityMapData(d_acti_map);
			startVisualisationLoop();
		}
	}
	else {
		double begin, end, elapsed;

		begin = omp_get_wtime();
		if (mode == 0) {
			serial_step();
		}
		else if (mode == 1) {
			openMP_step();
		}
		else {
			cuda_step();
		}
		end = omp_get_wtime();
		elapsed = end - begin;
		int second = (int)elapsed;
		int millisecond = (int)(1000.0f * (elapsed - second));
		printf("Execution time %d seconds %d milliseconds. \n", second, millisecond);
	}

	// function: free and unbind pointers.
	pointer_operation_end();
	
	return 0;
}
// command check
void command_check_NGM(int argc, char* argv[]) {
	if (argc < 4 || argc > 8) {
		if (strcmp(argv[1], "-help") == 0) {
			print_help();
			exit(1);
		}
		else {
			fprintf(stderr, "Error command (using -help). \n");
			exit(1);
		}
	}
	// n, g, m. n and g should be positive int. m should be OPENMP or CPU.
	if (!atoi(argv[1]) || !atoi(argv[2]) || atoi(argv[1]) <= 0 || atoi(argv[2]) <= 0 || !is_int(argv[1]) || !is_int(argv[2])) {
		fprintf(stderr, "Error: the first two arguments must be positive integer. \n");
		exit(1);
	}
	if ((strcmp(argv[3], "CPU") != 0) && (strcmp(argv[3], "OPENMP") != 0) && (strcmp(argv[3], "CUDA") != 0)) {
		fprintf(stderr, "Error: the third argument should be CPU or OPENMP or CUDA. \n");
		exit(1);
	}
	// if argc > 4, but there is no -i or -f next.
	if (argc > 4) {
		if ((strcmp(argv[4], "-i") != 0 && strcmp(argv[4], "-f") != 0) && argc != 4) {
			fprintf(stderr, "Error command (using -help). \n");
			exit(1);
		}
	}
}
void command_check_IF(int argc, char* argv[]) {
	int index_arg_i;
	for (int i = 0; i < argc; i++) {
		// -f. -f and filename should appear toghther.
		if (strcmp(argv[i], "-f") == 0) {
			if (i == argc - 1) {
				fprintf(stderr, "can't use -f without file_name.");
				error_func();
			}
			if (strcmp(argv[i + 1], "-i") == 0) {
				fprintf(stderr, "can't use -f without file_name.");
				error_func();
			}
			if (argv[i + 1] != NULL) {
				cond_arg_f = TRUE;
				index_arg_f = i + 1;
			}
			else {
				cond_arg_f = FALSE;
			}
		}
		// -i. I should be positive int & -i I should appear together.
		if (strcmp(argv[i], "-i") == 0) {
			if (i == argc - 1) {
				fprintf(stderr, "Error: can't use -i without I.");
				error_func();
			}
			if (strcmp(argv[i + 1], "-f") == 0) {
				fprintf(stderr, "Error: can't use -i without I.");
				error_func();
			}
			if (atoi(argv[i + 1])) {
				if (atoi(argv[i + 1]) < 0 || !is_int(argv[i + 1])) {
					fprintf(stderr, "Error: I must be positive int.");
					error_func();
				}
				cond_arg_i = TRUE;
				index_arg_i = i + 1;
			}
			else {
				fprintf(stderr, "Error command (using -help). \n");
				error_func();
			}
		}
	}
	if ((cond_arg_i && !cond_arg_f) || (!cond_arg_i && cond_arg_f)) {
		if (argc != 6) {
			fprintf(stderr, "Error command (using -help). \n");
			error_func();
		}
	}
	// iteration or visualiser.
	if (cond_arg_i) {
		iter = atoi(argv[index_arg_i]);
	}
	else {
		toggle = TRUE;
	}
}
void body_initializer(char* argv[]) {
	FILE* f = NULL;
	int k = 0; // index of heap.
	if (cond_arg_f) {
		f = fopen(argv[index_arg_f], "r");
		if (f == NULL) {
			fprintf(stderr, "Error: don't find file! \n");
			error_func();
		}
		// read format data from file.
		char str[30];
		while (fgets(str, 30, f) != NULL) {
			// avoid blank or comment.
			if (str[0] == '#' || str[0] == '\n' || str[0] == '\r') {
				continue;
			}
			else {
				if (k == num) {
					fprintf(stderr, "Error: file is bigger than allocated heap. \n");
					fclose(f);
					error_func();
				}
				char* ptr;
				char* tok;
				ptr = str;
				int index = 0;
				while ((tok = strsep(&ptr, ",")) != NULL) {
					if (strlen(tok) > 1) {
						// read from file.
						if (index == 0) body_pointer[k].x = (float)atof(tok);
						else if (index == 1) body_pointer[k].y = (float)atof(tok);
						else if (index == 2) body_pointer[k].vx = (float)atof(tok);
						else if (index == 3) body_pointer[k].vy = (float)atof(tok);
						else if (index == 4) body_pointer[k].m = (float)atof(tok);
						else fprintf(stderr, "Error: less than 5. \n");
					}
					else {
						// random.
						if (index == 0) body_pointer[k].x = rand() / (RAND_MAX + 1.0f);
						else if (index == 1) body_pointer[k].y = rand() / (RAND_MAX + 1.0f);
						else if (index == 2) body_pointer[k].vx = 0.0f;
						else if (index == 3) body_pointer[k].vy = 0.0f;
						else if (index == 4) body_pointer[k].m = 1.0f / num;
						else fprintf(stderr, "Error: less than 5. \n");
					}
					index++;
				}
				k++;
				index = 0;
			}
		}
		fclose(f);
	}
	else {
		for (int j = 0; j < num; j++) {
			body_pointer[k].x = rand() / (RAND_MAX + 1.0f);
			body_pointer[k].y = rand() / (RAND_MAX + 1.0f);
			body_pointer[k].vx = 0;
			body_pointer[k].vy = 0;
			body_pointer[k].m = 1.0f / num;
			k++;
		}
	}
	if (num != k) {
		fprintf(stderr, "num is not equal to nbodies in file. \n");
		error_func();
	}
}

// step function.
void serial_step(void) {
	float unit = 1.0f / grid;
	for (int k = 0; k < iter; k++) {
		memcpy(temp_pointer, body_pointer, sizeof(nbody) * num);
		for (int i = 0; i < (grid * grid); i++) { 
			acti_map[i] = 0.0f; 
		}
		// when i == j, F is zero. 
		for (int i = 0; i < num; i++) {
			float force_x = 0.0f;
			float force_y = 0.0f;
			float body_x = temp_pointer[i].x;
			float body_y = temp_pointer[i].y;
			float body_m = temp_pointer[i].m;
			for (int j = 0; j < num; j++) {
				float dis_x = temp_pointer[j].x - body_x;
				float dis_y = temp_pointer[j].y - body_y;
				float magnitude = (float) sqrt((double) dis_x*dis_x + (double) dis_y*dis_y);
				force_x += (temp_pointer[j].m*dis_x) / (float) pow(((double) magnitude + (double) SOFTENING*SOFTENING), 3.0 / 2);
				force_y += (temp_pointer[j].m*dis_y) / (float) pow(((double) magnitude + (double) SOFTENING*SOFTENING), 3.0 / 2);
			}
			force_x *= G * body_m;
			force_y *= G * body_m;
			float acc_x = force_x / body_m;
			float acc_y = force_y / body_m;
			float new_x = body_x + dt * temp_pointer[i].vx;
			float new_y = body_y + dt * temp_pointer[i].vy;
			body_pointer[i].vx = temp_pointer[i].vx + dt * acc_x;
			body_pointer[i].vy = temp_pointer[i].vy + dt * acc_y;
			body_pointer[i].x = new_x;
			body_pointer[i].y = new_y;

			// update density.
			if (new_x < 0 || new_x >1 || new_x < 0 || new_y > 1) continue;
			int index_y = (int)floor(new_y / unit);
			int index_x = (int)floor(new_x / unit);
			int index = grid * index_y + index_x;
			acti_map[index] += (1.0f / num * 10.0f);
		}
	}
}
void openMP_step(void) {
	float unit = 1.0f / grid;
	omp_set_nested(1);
	int k;
	#pragma omp parallel for
	for (k = 0; k < iter; k++) {
		memcpy(temp_pointer, body_pointer, sizeof(nbody) * num);
		for (int i = 0; i < (grid * grid); i++) acti_map[i] = 0.0f;
		// when i == j, F is zero. 
		#pragma omp parallel default(none)
		{
			int i;
			#pragma omp for schedule(dynamic)
			for (i = 0; i < num; i++) {
				float force_x = 0.0f;
				float force_y = 0.0f;
				for (int j = 0; j < num; j++) {
					if (i == j) continue;
					float dis_x = temp_pointer[j].x - temp_pointer[i].x;
					float dis_y = temp_pointer[j].y - temp_pointer[i].y;
					float magnitude = (float)sqrt((double)dis_x * dis_x + (double)dis_y * dis_y);
					force_x += (temp_pointer[j].m * dis_x) / (float)pow(((double)magnitude + (double)SOFTENING * SOFTENING), 3.0 / 2);
					force_y += (temp_pointer[j].m * dis_y) / (float)pow(((double)magnitude + (double)SOFTENING * SOFTENING), 3.0 / 2);
				}

				force_x *= G * temp_pointer[i].m;
				force_y *= G * temp_pointer[i].m;
				float new_x = temp_pointer[i].x + dt * temp_pointer[i].vx;
				float new_y = temp_pointer[i].y + dt * temp_pointer[i].vy;
				body_pointer[i].vx = temp_pointer[i].vx + dt * (force_x / temp_pointer[i].m);
				body_pointer[i].vy = temp_pointer[i].vy + dt * (force_y / temp_pointer[i].m);
				body_pointer[i].x = new_x;
				body_pointer[i].y = new_y;

				// update density.
				if (new_x < 0 || new_x >1 || new_x < 0 || new_y > 1) continue;
				int index_y = (int)floor(new_y / unit);
				int index_x = (int)floor(new_x / unit);
				int index = grid * index_y + index_x;
				#pragma omp atomic
				acti_map[index] += (1.0f / num * 10.0f);
			}
		}
	}
}

void print_help(){
	printf("nbody_%s N D M [-i I] [-i input_file]\n", USER_NAME);

	printf("where:\n");
	printf("\tN                Is the number of bodies to simulate.\n");
	printf("\tD                Is the integer dimension of the activity grid. The Grid has D*D locations.\n");
	printf("\tM                Is the operation mode, either  'CPU' or 'OPENMP'\n");
	printf("\t[-i I]           Optionally specifies the number of simulation iterations 'I' to perform. Specifying no value will use visualisation mode. \n");
	printf("\t[-f input_file]  Optionally specifies an input file with an initial N bodies of data. If not specified random data will be created.\n");
}
void error_func() {
	free(body_pointer);
	free(acti_map);
	exit(1);
}
char* strsep(char** stringp, const char* delim) {
	char* s;
	const char* spanp;
	int c, sc;
	char* tok;

	if ((s = *stringp) == NULL)
		return (NULL);
	for (tok = s;;) {
		c = *s++;
		spanp = delim;
		do {
			if ((sc = *spanp++) == c) {
				if (c == 0)
					s = NULL;
				else
					s[-1] = 0;
				*stringp = s;
				return (tok);
			}
		} while (sc != 0);
	}
	/* NOTREACHED */
}
boolean is_int(char* str) {
	boolean is_int = TRUE;
	for (int i = 0; i < (signed int) strlen(str); i++) {
		if (str[i] == '.') is_int = FALSE;
	}
	return is_int;
}

void pointer_operation_start() {
	// allocate device memory.
	cudaMalloc((void**)&d_body_pointer, sizeof(struct nbody) * num);
	cudaMalloc((void**)&d_temp_pointer, sizeof(struct nbody) * num);
	cudaMalloc((void**)&d_acti_map, sizeof(float) * (grid * grid));

	// copy data from host to device. nbody, num, grid.
	cudaMemcpy(d_body_pointer, body_pointer, sizeof(struct nbody) * num, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(nbody_count, &num, sizeof(int));
	cudaMemcpyToSymbol(nbody_grid, &grid, sizeof(int));

	// d_soa_body_pointer will not be update within force calculation.
	cudaMalloc((void**)&d_soa_body_pointer.x, sizeof(float) * num);
	cudaMalloc((void**)&d_soa_body_pointer.y, sizeof(float) * num);
	cudaMalloc((void**)&d_soa_body_pointer.vx, sizeof(float) * num);
	cudaMalloc((void**)&d_soa_body_pointer.vy, sizeof(float) * num);
	cudaMalloc((void**)&d_soa_body_pointer.m, sizeof(float) * num);

	// d_soa_temp_pointer will be update within force calculation.
	cudaMalloc((void**)&d_soa_temp_pointer.x, sizeof(float) * num);
	cudaMalloc((void**)&d_soa_temp_pointer.y, sizeof(float) * num);
	cudaMalloc((void**)&d_soa_temp_pointer.vx, sizeof(float) * num);
	cudaMalloc((void**)&d_soa_temp_pointer.vy, sizeof(float) * num);
	cudaMalloc((void**)&d_soa_temp_pointer.m, sizeof(float) * num);

	cudaBindTexture(0, arr_x, d_soa_body_pointer.x, sizeof(float) * num);
	cudaBindTexture(0, arr_y, d_soa_body_pointer.y, sizeof(float) * num);
	cudaBindTexture(0, arr_vx, d_soa_body_pointer.vx, sizeof(float) * num);
	cudaBindTexture(0, arr_vy, d_soa_body_pointer.vy, sizeof(float) * num);
	cudaBindTexture(0, arr_m, d_soa_body_pointer.m, sizeof(float) * num);
}
void pointer_operation_end() {

	cudaFree(d_body_pointer);
	cudaFree(d_temp_pointer);
	cudaFree(d_acti_map);

	cudaUnbindTexture(arr_x);
	cudaUnbindTexture(arr_y);
	cudaUnbindTexture(arr_vx);
	cudaUnbindTexture(arr_vy);
	cudaUnbindTexture(arr_m);

	cudaFree(d_soa_body_pointer.x);
	cudaFree(d_soa_body_pointer.y);
	cudaFree(d_soa_body_pointer.vx);
	cudaFree(d_soa_body_pointer.vy);
	cudaFree(d_soa_body_pointer.m);

	cudaFree(d_soa_temp_pointer.x);
	cudaFree(d_soa_temp_pointer.y);
	cudaFree(d_soa_temp_pointer.vx);
	cudaFree(d_soa_temp_pointer.vy);
	cudaFree(d_soa_temp_pointer.m);

	free(body_pointer);
	free(temp_pointer);
	free(acti_map);
}
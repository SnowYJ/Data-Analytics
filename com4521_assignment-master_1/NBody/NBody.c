#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#include "NBody.h"
#include "NBodyVisualiser.h"

#define USER_NAME "acz19yz"

void print_help();

// serial.
void step(void);
void acti_map_func(float *acti_map);

// openMP
void openMP_step(void);
void openMP_acti_map_func(float *acti_map);

// N, D, M and I.
int num; 
int grid; 
MODE mode; 
int iter = 1; 

// pointer: acti_map: density, body_pointer: struct body.
nbody *body_pointer; 
const float *acti_map;

// those variables are for calculating the boundary.
float max_x = 0.0;
float max_y = 0.0; 
float min_x;
float min_y;

// argument 'b' and 'd'.
boolean toggle = FALSE;

// initialize boundary.
boolean cond1 = TRUE;

int main(int argc, char *argv[]) {
	FILE *f = NULL;
	int k = 0; // index of heap.
	// argument -i -f.
	boolean cond_arg_i = FALSE;
	boolean cond_arg_f = FALSE;
	int index_arg_f, index_arg_i;

	// help.
	if (argc < 4 || argc > 8) {
		if (strcmp(argv[1], "-help") == 0) {
			print_help();
			exit(1);
		}
		else {
			fprintf(stderr, "Error command. \n");
			exit(1);
		}
	}
	// n, g, m.
	if (!atoi(argv[1]) || !atoi(argv[2]) || atoi(argv[1])<=0 || atoi(argv[2])<=0) {
		fprintf(stderr, "Error: the first two arguments must be positive integer. \n");
		exit(1);
	}
	if ((strcmp(argv[3], "CPU") != 0) && (strcmp(argv[3], "OPENMP") != 0)) {
		fprintf(stderr, "Error: the third argument should be CPU or OPENMP. \n");
		exit(1);
	}
	num = atoi(argv[1]);
	grid = atoi(argv[2]);
	mode = (strcmp(argv[3], "CPU") == 0) ? 0 : 1;

	//TODO: Allocate any heap memory.
	body_pointer = (struct vec *) malloc(sizeof(struct nbody) * num);
	// allocate heap for recording density.
	acti_map = (float *)malloc(sizeof(float)*(grid * grid));
	// Can't not initialize float array using memset() !!!
	
	// -i & -f. whether command has -i -f.
	for (int i = 0; i < argc; i++) {
		// -f.
		if (strcmp(argv[i], "-f") == 0) {
			if (i == argc - 1) {
				fprintf(stderr, "can't use -f without file_name.");
				exit(1);
			}
			if (argv[i + 1]!=NULL) {
				cond_arg_f = TRUE;
				index_arg_f = i + 1;
			}
			else {
				cond_arg_f = FALSE;
			}
		}
		// -i.
		if (strcmp(argv[i], "-i") == 0) {
			if (i == argc - 1) {
				fprintf(stderr, "Error: can't use -i without I.");
				exit(1);
			}
			if (atoi(argv[i + 1])) {
				if (atoi(argv[i + 1]) < 0) {
					fprintf(stderr, "Error: I must be positive int.");
					exit(1);
				}
				cond_arg_i = TRUE;
				index_arg_i = i + 1;
			}
			else {
				fprintf(stderr, "Error: can't use -i without I.");
				exit(1);
			}
		}
	}
	// iteration.
	if (cond_arg_i) {
		iter = atoi(argv[index_arg_i]);
	}
	else {
		// visualiser.
		toggle = TRUE;
		printf("Running visualiser. \n");
	}
	// read file.
	if (cond_arg_f) {
		f = fopen(argv[index_arg_f], "r");
		if (f == NULL) {
			fprintf(stderr, "Error: don't find file! \n");
			exit(1);
		}
		int len;
		// read format data from file.
		while ((len = fscanf(f, "%f, %f, %f, %f, %f", &body_pointer[k].x, &body_pointer[k].y, &body_pointer[k].vx, &body_pointer[k].vy, &body_pointer[k].m)) != EOF) {
			if (len == 0) {
				fscanf(f, "%*[^\n]%*"); // skip a comment line.
				continue;
			}
			// fill with default value.
			if ((len != 5) && (len != 0)) {
				if (len == 1) { body_pointer[k].y = rand() / (RAND_MAX + 1.0); len++; }
				if (len == 2) { body_pointer[k].vx = 0; len++; }
				if (len == 3) { body_pointer[k].vy = 0; len++; }
				if (len == 4) { body_pointer[k].m = 1.0 / num; len++; }
			}
			k++;
			if (k > num) {
				fprintf(stderr, "Error: num should bigger than amount of nbody in file.");
				exit(1);
			}
		}
		if (k != num) {
			fprintf(stderr, "Error: nbodies are not the same. \n");
			free(body_pointer);
			free(acti_map);
			exit(1);
		}
		fclose(f);
	}
	else {
		for (int j = 0; j < num; j++) {
			body_pointer[k].x = rand() / (RAND_MAX + 1.0);
			body_pointer[k].y = rand() / (RAND_MAX + 1.0);
			body_pointer[k].vx = 0;
			body_pointer[k].vy = 0;
			body_pointer[k].m = 1.0 / num;
			k++;
		}
	}

	// visualizer or iteration simulation.
	if (toggle) {
		if (mode == 0) {
			initViewer(num, grid, mode, &step);
		}
		else {
			initViewer(num, grid, mode, &openMP_step);
		}
		setNBodyPositions(body_pointer);
		setHistogramData(acti_map);
		startVisualisationLoop(); 
	}
	else {
		double begin, end;
		double elapsed;
		begin = omp_get_wtime();
		if (mode == 0) {
			step();
		}
		else {
			openMP_step();
		}
		end = omp_get_wtime();
		elapsed = end - begin;
		int second = (int)elapsed;
		int millisecond = 1000 * (elapsed - second);
		printf("Execution time %d seconds %d milliseconds. \n", second, millisecond);
	}

	printf("Do you want to output the body position (using b) or heap map (using d) or (any other key to exit): ");
	char c;
	while ((c = getchar()) != 'n') {
		if (c == 'b') {
			for (int i = 0; i < num; i++) {
				printf("this is body %d \n", i + 1);
				printf("%f \n", body_pointer[i].x);
				printf("%f \n", body_pointer[i].y);
				printf("%f \n", body_pointer[i].vx);
				printf("%f \n", body_pointer[i].vy);
				printf("%f \n", body_pointer[i].m);
			}
		}
		else if (c == 'd') {
			for (int i = 0; i < grid*grid; i++) {
				printf("the %d index, value : %.4f . \n", i, acti_map[i]); 
			}
		}
		else {
			break;
		}
	}

	free(body_pointer);
	free(acti_map);
	return 0;
}

float left_bound, right_bound, top_bound, bottom_bound, unit;
// CPU function.
void step(void) {
	//TODO: Perform the main simulation of the NBody system.
	boolean cond = TRUE;
	int i, j; // index of body.
	int k = 0;
	for (int k = 0; k < iter; k++) {
		// when i == j, F is zero. 
		for (i = 0; i < num; i++) {
			float force_x = 0.0f;
			float force_y = 0.0f;
			for (j = 0; j < num; j++) {
				float dis_x = body_pointer[j].x - body_pointer[i].x;
				float dis_y = body_pointer[j].y - body_pointer[i].y;
				float magnitude = sqrt(dis_x*dis_x + dis_y * dis_y);
				force_x += (G * body_pointer[i].m*body_pointer[j].m*dis_x) / pow((magnitude + SOFTENING), 3.0 / 2);
				force_y += (G * body_pointer[i].m*body_pointer[j].m*dis_y) / pow((magnitude + SOFTENING), 3.0 / 2);
			}
			float acc_x = force_x / body_pointer[i].m;
			float acc_y = force_y / body_pointer[i].m;
			body_pointer[i].vx = body_pointer[i].vx + dt * acc_x;
			body_pointer[i].vy = body_pointer[i].vy + dt * acc_y;
			body_pointer[i].x = body_pointer[i].x + dt * body_pointer[i].vx;
			body_pointer[i].y = body_pointer[i].y + dt * body_pointer[i].vy;

			// deciding the boundary. the boundary will be fixed according to the position of Nbody system after the first iteration.
			if (cond) {
				max_x = body_pointer[i].x;
				min_x = body_pointer[i].x;
				max_y = body_pointer[i].y;
				min_y = body_pointer[i].y;
				cond = FALSE;
			}

			if (body_pointer[i].x > max_x) {
				max_x = body_pointer[i].x;
			}
			if (body_pointer[i].y > max_y) {
				max_y = body_pointer[i].y;
			}
			if (body_pointer[i].x < min_x) {
				min_x = body_pointer[i].x;
			}
			if (body_pointer[i].y < min_y) {
				min_y = body_pointer[i].y;
			}
		}
		acti_map_func(acti_map);
	}
}
void acti_map_func(float *acti_map) {
	// initialize.
	for (int i = 0; i < (grid*grid); i++) {
		acti_map[i] = 0.0f;
	}
	// first iteration useful.
	if (cond1) {
		// get the center of N-body.
		float cen_x = (max_x - min_x) / 2.0;
		float cen_y = (max_y - min_y) / 2.0;
		// spread the boundary according to center. eg. center +- 300.
		left_bound = cen_x - 1.0;
		right_bound = cen_x + 1.0;
		top_bound = cen_y + 1.0;
		bottom_bound = cen_y - 1.0;
		// split square into grid.
		unit = 2.0 / grid;
		cond1 = FALSE;
	}

	for (int i = 0; i < num; i++) {
		float temp_x = body_pointer[i].x;
		float temp_y = body_pointer[i].y;
		if (temp_x < left_bound || temp_x > right_bound || temp_y < bottom_bound || temp_y > top_bound) {
			continue;
		}
		else {
			int index_y = floor((temp_y - bottom_bound) / unit);
			int index_x = floor((temp_x - left_bound) / unit);
			// put into array.
			int index = grid * index_y + index_x;
			acti_map[index] += 1.0f;
		}
	}
	// normalize density.
	for (int i = 0; i < (grid*grid); i++) {
		acti_map[i] = acti_map[i] / num;
	}
}

// openMP function.
void openMP_acti_map_func(float *acti_map) {
	// initialize.
	int i;
	#pragma omp parallel for 
	for (i = 0; i < (grid*grid); i++) {
		acti_map[i] = 0.0f;
	}
	// first iteration useful.
	if (cond1) {
		// get the center of N-body.
		float cen_x = (max_x - min_x) / 2.0;
		float cen_y = (max_y - min_y) / 2.0;
		// spread the boundary according to center. eg. center +- 300.
		left_bound = cen_x - 1.0;
		right_bound = cen_x + 1.0;
		top_bound = cen_y + 1.0;
		bottom_bound = cen_y - 1.0;
		// split square into grid.
		unit = 2.0 / grid;
		cond1 = FALSE;
	}
	int j;
	#pragma omp parallel for
	for (j = 0; j < num; j++) {
		float temp_x = body_pointer[j].x;
		float temp_y = body_pointer[j].y;
		if (temp_x < left_bound || temp_x > right_bound || temp_y < bottom_bound || temp_y > top_bound) {
			continue;
		}
		else {
			int index_y = floor((temp_y - bottom_bound) / unit);
			int index_x = floor((temp_x - left_bound) / unit);
			// put into array.
			int index = grid * index_y + index_x;
			acti_map[index] += 1.0f;
		}
	}
	// normalize density.
	int k;
	#pragma omp parallel for
	for (k = 0; k < (grid*grid); k++) {
		acti_map[k] = acti_map[k] / num;
	}
}
void openMP_step(void) {
	//TODO: Perform the main simulation of the NBody system.
	boolean cond = TRUE;
	int k;
	omp_set_nested(1);
	#pragma omp parallel for
	for (k = 0; k < iter; k++) {
		// when i == j, F is zero. 
		int i; 
		#pragma omp parallel for
		for (i = 0; i < num; i++) {
			float force_x = 0.0f;
			float force_y = 0.0f;
			int j;
			for (j = 0; j < num; j++) {
				float dis_x = body_pointer[j].x - body_pointer[i].x;
				float dis_y = body_pointer[j].y - body_pointer[i].y;
				float magnitude = sqrt(dis_x*dis_x + dis_y * dis_y);
				force_x += (G * body_pointer[i].m*body_pointer[j].m*dis_x) / pow((magnitude + SOFTENING), 3.0 / 2);
				force_y += (G * body_pointer[i].m*body_pointer[j].m*dis_y) / pow((magnitude + SOFTENING), 3.0 / 2);
			}
			float acc_x = force_x / body_pointer[i].m;
			float acc_y = force_y / body_pointer[i].m;
			body_pointer[i].vx = body_pointer[i].vx + dt * acc_x;
			body_pointer[i].vy = body_pointer[i].vy + dt * acc_y;
			body_pointer[i].x = body_pointer[i].x + dt * body_pointer[i].vx;
			body_pointer[i].y = body_pointer[i].y + dt * body_pointer[i].vy;

			// deciding the boundary. the boundary will be fixed according to the position of Nbody system after the first iteration.
			if (cond) {
				max_x = body_pointer[i].x;
				min_x = body_pointer[i].x;
				max_y = body_pointer[i].y;
				min_y = body_pointer[i].y;
				cond = FALSE;
			}

			if (body_pointer[i].x > max_x) {
				max_x = body_pointer[i].x;
			}
			if (body_pointer[i].y > max_y) {
				max_y = body_pointer[i].y;
			}
			if (body_pointer[i].x < min_x) {
				min_x = body_pointer[i].x;
			}
			if (body_pointer[i].y < min_y) {
				min_y = body_pointer[i].y;
			}
		}
		acti_map_func(acti_map);
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

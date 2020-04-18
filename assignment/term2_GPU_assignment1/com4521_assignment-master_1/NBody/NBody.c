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

// strsep
char* strsep(char** stringp, const char* delim);
// error
void error_func();
// is integer
boolean is_int(char* str);
// serial
void step(void);
void acti_map_func();
// openMP
void openMP_step(void);
void openMP_acti_map_func();
// N, D, M and I.
int num; 
int grid; 
MODE mode; 
int iter = 1; 
// pointer: acti_map: density, body_pointer: struct body.
nbody *body_pointer; 
float *acti_map;
// those variables are for calculating the boundary in step().
float max_x = 0.0;
float max_y = 0.0; 
float min_x;
float min_y;
// running visualizer if true.
boolean toggle = FALSE;
// initialize boundary.
boolean cond1 = TRUE;

/*
Steps:
1. check command n, g, m
2. allocate heap and initialization n, g, m.
3. check command -i, -f.
4. read file or random initialization & toggle of visualizer or iteration simuliation.
5. running visualizer or iteration.
*/
int main(int argc, char *argv[]) {
	FILE *f = NULL;
	int k = 0; // index of heap.
	// command has argument -i or -f.
	boolean cond_arg_i = FALSE;
	boolean cond_arg_f = FALSE;
	// index of I or file_name.
	int index_arg_f, index_arg_i;
	// ---------------------------------------------- check command n, g, m -------------------------------------------------------
	// help.
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
	if (!atoi(argv[1]) || !atoi(argv[2]) || atoi(argv[1])<=0 || atoi(argv[2])<=0 || !is_int(argv[1]) || !is_int(argv[2])) {
		fprintf(stderr, "Error: the first two arguments must be positive integer. \n");
		exit(1);
	}
	if ((strcmp(argv[3], "CPU") != 0) && (strcmp(argv[3], "OPENMP") != 0)) {
		fprintf(stderr, "Error: the third argument should be CPU or OPENMP. \n");
		exit(1);
	}
	// if argc > 4, but there is no -i or -f next.
	if (argc > 4) {
		if ((strcmp(argv[4], "-i") != 0 && strcmp(argv[4], "-f") != 0) && argc != 4) {
			fprintf(stderr, "Error command (using -help). \n");
			exit(1);
		}
	}
	// --------------------------------- n, g, m initialization & allocate heap ----------------------------------------------
	num = atoi(argv[1]);
	grid = atoi(argv[2]);
	mode = (strcmp(argv[3], "CPU") == 0) ? 0 : 1;

	// allocate heap of n-body.
	body_pointer = (struct nbody *) malloc(sizeof(struct nbody) * num);
	// allocate heap for recording density.
	acti_map = (float *)malloc(sizeof(float)*(grid * grid));

	// -------------------------------------------------- check command -i -f ------------------------------------------------ 
	// from now on, body_pointer and acti_map should be free when error happened.
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
				if (atoi(argv[i + 1]) < 0 || !is_int(argv[i+1])) {
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

	// ------------------------------------------- read file or random initialization ----------------------------------------
	if (cond_arg_f) {
		f = fopen(argv[index_arg_f], "r");
		if (f == NULL) {
			fprintf(stderr, "Error: don't find file! \n");
			error_func();
		}
		// read format data from file.
		char str[30];
		int index = 0;
		while (fgets(str, 30, f) != NULL) {
			// avoid blank or comment.
			if (str[0] == '#'|| str[0] == '\n' || str[0] == '\r') {
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
						if (index == 0) body_pointer[k].x = (float) atof(tok);
						else if (index == 1) body_pointer[k].y = (float) atof(tok);
						else if (index == 2) body_pointer[k].vx = (float) atof(tok);
						else if (index == 3) body_pointer[k].vy = (float) atof(tok);
						else if (index == 4) body_pointer[k].m = (float) atof(tok);
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


	// ---------------------------------------- running visualizer or iteration simulation -----------------------------------
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
		int second = (int) elapsed;
		int millisecond = (int) (1000.0f * (elapsed - second));
		printf("Execution time %d seconds %d milliseconds. \n", second, millisecond);
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
				float magnitude = (float) sqrt((double) dis_x*dis_x + (double) dis_y*dis_y);
				force_x += (G * body_pointer[i].m*body_pointer[j].m*dis_x) / (float) pow(((double) magnitude + (double) SOFTENING), 3.0 / 2);
				force_y += (G * body_pointer[i].m*body_pointer[j].m*dis_y) / (float) pow(((double) magnitude + (double) SOFTENING), 3.0 / 2);
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
		acti_map_func();
	}
}
void acti_map_func() {
	// initialize. Can't not initialize float array using memset() !!!
	for (int i = 0; i < (grid*grid); i++) {
		acti_map[i] = 0.0f;
	}
	// only useful in first iteration.
	if (cond1) {
		// get the center of N-body.
		float cen_x = (max_x - min_x) / 2.0f;
		float cen_y = (max_y - min_y) / 2.0f;
		// spread the boundary according to center. eg. center +- 0.5.
		left_bound = cen_x - 0.5f;
		right_bound = cen_x + 0.5f;
		top_bound = cen_y + 0.5f;
		bottom_bound = cen_y - 0.5f;
		// split square into grid.
		unit = 1.0f / grid;
		cond1 = FALSE;
	}
	for (int i = 0; i < num; i++) {
		float temp_x = body_pointer[i].x;
		float temp_y = body_pointer[i].y;
		if (temp_x < left_bound || temp_x > right_bound || temp_y < bottom_bound || temp_y > top_bound) {
			continue;
		}
		else {
			int index_y = (int) floor((temp_y - bottom_bound) / unit);
			int index_x = (int) floor((temp_x - left_bound) / unit);
			// put into array.
			int index = grid * index_y + index_x;
			acti_map[index] += 1.0f;
		}
	}
	// normalize density.
	for (int i = 0; i < (grid*grid); i++) {
		acti_map[i] = acti_map[i] / num * 10.0f;
	}
}

// openMP function.
void openMP_step(void) {
	//TODO: Perform the main simulation of the NBody system.
	boolean cond = TRUE;
	int k;
	for (k = 0; k < iter; k++) {
		// when i == j, F is zero. 
		int i;
		#pragma omp parallel for schedule(dynamic)
		for (i = 0; i < num; i++) {
			float force_x = 0.0f;
			float force_y = 0.0f;
			int j;
			for (j = 0; j < num; j++) {
				float dis_x = body_pointer[j].x - body_pointer[i].x;
				float dis_y = body_pointer[j].y - body_pointer[i].y;
				float magnitude = (float)sqrt((double)dis_x * dis_x + (double)dis_y * dis_y);
				force_x += (G * body_pointer[i].m * body_pointer[j].m * dis_x) / (float)pow(((double)magnitude + (double)SOFTENING), 3.0 / 2);
				force_y += (G * body_pointer[i].m * body_pointer[j].m * dis_y) / (float)pow(((double)magnitude + (double)SOFTENING), 3.0 / 2);
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
		openMP_acti_map_func();
	}
}
void openMP_acti_map_func() {
	// initialization.
	int i;
	#pragma omp parallel for
	for (i = 0; i < (grid*grid); i++) {
		acti_map[i] = 0.0f;
	}
	#pragma omp barrier
	// first iteration useful.
	if (cond1) {
		// get the center of N-body.
		float cen_x = (max_x - min_x) / 2.0f;
		float cen_y = (max_y - min_y) / 2.0f;
		// spread the boundary according to center. eg. center +- 0.5.
		left_bound = cen_x - 0.5f;
		right_bound = cen_x + 0.5f;
		top_bound = cen_y + 0.5f;
		bottom_bound = cen_y - 0.5f;
		// split square into grid.
		unit = 1.0f / grid;
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
			int index_y = (int) floor((temp_y - bottom_bound) / unit);
			int index_x = (int) floor((temp_x - left_bound) / unit);
			// put into array.
			int index = grid * index_y + index_x;
			acti_map[index] += 1.0f;
		}
	}
	#pragma omp barrier
	int k;
	#pragma omp parallel for
	// normalize density.
	for (k = 0; k < (grid*grid); k++) {
		acti_map[k] = acti_map[k] / num * 10.0f;
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
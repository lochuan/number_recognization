#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>

#define RESOLUTION 28  // bmp image 28*28
#define LINE_LEN (RESOLUTION * RESOLUTION * 4) // All the pixel data stored within a line
#define DATA_NUM_WITHOUT_LABEL (RESOLUTION * RESOLUTION) // Data length without label
#define DATA_NUM_WITH_LABEL ((RESOLUTION * RESOLUTION) + 1) // For training, the first number is label, so +1
#define HIDDEN_NODE 250 // node number in hidden layer
#define OUTPUT_NODE 10  // output node number
#define LEARN_RATE 0.1
#define TRAIN_SET_SIZE 5000 
#define TEST_SET_SIZE 1000
#define EPOCH 8 

#define SIGMOID(x) (1.0 / (1.0 + (pow(M_E, -x))))
#define NOR_DIS_DEV(x) (pow(x, -0.5))  // For generating init weight

FILE* openfile(char* file_name);
void read_parse(FILE* fp, int input_data[], int* label, char* mode);
gsl_matrix* scale_input(int input_data[]);
gsl_matrix* vectorize_target(int label);
gsl_matrix* init_weight(int row, int col);
double get_output_error(gsl_matrix* target, gsl_matrix* output, gsl_matrix* output_error);
void training(gsl_matrix* better_input, gsl_matrix* target, gsl_matrix* input_weight, gsl_matrix* hidden_weight, gsl_matrix* hidden_x, gsl_matrix* hidden_o, gsl_matrix* output_o, gsl_matrix* output_x, gsl_matrix* output_error, gsl_matrix* hidden_error, double* error);
void store_weight(gsl_matrix* input_weight, gsl_matrix* hidden_weight);
void load_weight(gsl_matrix* input_weight, gsl_matrix* hidden_weight);
int recognize(gsl_matrix* better_input, gsl_matrix* input_weight, gsl_matrix* hidden_weight, gsl_matrix* hidden_x, gsl_matrix* hidden_o, gsl_matrix* output_x, gsl_matrix* output_o);

int main(int argc, char **argv)
{
	int label;
	double error;
	double error_acc = 0.0;
	char* file_name;
	char* mode;
	FILE* fp;
	int input_data[DATA_NUM_WITHOUT_LABEL];
	gsl_matrix* better_input;
	gsl_matrix* target;
	gsl_matrix* output_error = gsl_matrix_alloc(OUTPUT_NODE, 1);
	gsl_matrix* hidden_error = gsl_matrix_alloc(HIDDEN_NODE, 1);
	gsl_matrix* output_x = gsl_matrix_alloc(OUTPUT_NODE, 1);
	gsl_matrix* output_o = gsl_matrix_alloc(OUTPUT_NODE, 1);
    gsl_matrix* hidden_x = gsl_matrix_alloc(HIDDEN_NODE, 1);
    gsl_matrix* hidden_o = gsl_matrix_alloc(HIDDEN_NODE, 1);
	gsl_matrix* input_weight;
	gsl_matrix* hidden_weight;

	if(argc < 3){
		fprintf(stderr, "Usage: %s [-t(raining), -r(cognizing), -e(poch)] [file_name]\n", argv[0]);
		exit(EXIT_FAILURE);
	}else{
		mode = argv[1];
		file_name = argv[2];
	}

	fp = openfile(file_name);

	if(!strcmp(mode, "-t")){
		//training
		input_weight = init_weight(HIDDEN_NODE, DATA_NUM_WITHOUT_LABEL); // initiate the input-hidden weight
    	hidden_weight = init_weight(OUTPUT_NODE, HIDDEN_NODE);  // initiate the hidden-output weight

    	int training_times = 0;
    	puts("Training started\n");
        for (int i = 0; i < EPOCH; i++){

			do{
				read_parse(fp, input_data, &label, mode);
				better_input = scale_input(input_data);
				target = vectorize_target(label);
				training(better_input, target, input_weight, hidden_weight, hidden_x, hidden_o, output_o, output_x, output_error, hidden_error, &error);
				training_times++;
                error_acc += error;

			}while(training_times < TRAIN_SET_SIZE);
			
			rewind(fp);  // back to the start of the input file, start epoch
			if(i == 0){
				printf("In the first training, Accumulative Error = %f\n", error_acc);
				puts("========Start Epoch=========");
			}else{
				printf("Epoch %d, Accumulative Error = %f\n", i, error_acc);
			}
			error_acc = 0.0;
            training_times = 0;

        }
		
        gsl_matrix_free(better_input), gsl_matrix_free(target), gsl_matrix_free(output_error), gsl_matrix_free(hidden_error), gsl_matrix_free(output_x), gsl_matrix_free(output_o);
        gsl_matrix_free(hidden_x), gsl_matrix_free(hidden_o);

        store_weight(input_weight, hidden_weight);
        puts("Training weight has been stored!");
        gsl_matrix_free(input_weight), gsl_matrix_free(hidden_weight);

	}
	

	if(!strcmp(mode, "-r")){
		input_weight = gsl_matrix_alloc(HIDDEN_NODE, DATA_NUM_WITHOUT_LABEL);
		hidden_weight = gsl_matrix_alloc(OUTPUT_NODE, HIDDEN_NODE);

		load_weight(input_weight, hidden_weight);

		int test_times = 0;
		int wrong_times = 0;
		int anwser;

		do{
            read_parse(fp, input_data, &label, mode);
			better_input = scale_input(input_data);
			anwser = recognize(better_input, input_weight, hidden_weight, hidden_x, hidden_o, output_x, output_o);

			printf("Label: %d, Guess: %d\n", label, anwser);

			if(anwser != label) wrong_times++;

			test_times++;

		}while(test_times < TEST_SET_SIZE);
		printf("Performance: %2f\n", (1 - ((float)wrong_times/(float)test_times)) * 100);
		gsl_matrix_free(better_input), gsl_matrix_free(target), gsl_matrix_free(output_error), gsl_matrix_free(hidden_error), gsl_matrix_free(output_x), gsl_matrix_free(output_o);
		gsl_matrix_free(hidden_x), gsl_matrix_free(hidden_o);
		gsl_matrix_free(input_weight), gsl_matrix_free(hidden_weight);

	}
        fclose(fp);
		return 0;
}

FILE* openfile(char* file_name)
{
	FILE* fp;
	if(!(fp = fopen(file_name, "r")))
		fprintf(stderr, "Can't open the file\n");
	return fp;
}

void read_parse(FILE* fp, int input_data[], int* label, char* mode)
{
	char line[LINE_LEN];
	char temp;
	int line_c = 0;
	int input_c = 0;
	char* token;

	while((temp = fgetc(fp)) != '\n'){
        if(temp == EOF) break;
        line[line_c++] = temp;
	}
    line[line_c] = '\0';
    line_c = 0;
	if(!strcmp(mode, "-t")){
		token = strtok(line, ",");
		*label = atoi(token);
		while(token != NULL){
			token = strtok(NULL, ",");
			if(token == NULL) break;
			input_data[input_c++] = atoi(token);
		}

	}
	if(!strcmp(mode, "-r")){
		token = strtok(line, ",");
		*label = atoi(token);
		while(token != NULL){
			token = strtok(NULL, ",");
			if(token == NULL) break;
			input_data[input_c++] = atoi(token);
		}
	}
}

gsl_matrix* scale_input(int input_data[])
{
    gsl_matrix* better_input = gsl_matrix_alloc(DATA_NUM_WITHOUT_LABEL, 1);
	for(int i = 0; i < DATA_NUM_WITHOUT_LABEL; i++){
		gsl_matrix_set(better_input, i, 0, ((double)input_data[i]/255.0*0.99)+0.01);
	}
    return better_input;
}

gsl_matrix* vectorize_target(int label)
{
	gsl_matrix* target = gsl_matrix_alloc(10, 1);

	for(int i = 0; i < 10; i++){
		if(i == label){
			gsl_matrix_set(target, i, 0, 0.99);
		}else{
			gsl_matrix_set(target, i, 0, 0.01);
		}
	}

	return target;
}

gsl_matrix* init_weight(int row, int col)
{
    gsl_matrix* input_weight = gsl_matrix_alloc(row, col);

    const gsl_rng_type * T;
    gsl_rng * r;
    double sigma = NOR_DIS_DEV(col);
    r = gsl_rng_alloc (gsl_rng_mt19937);
    for(int i = 0; i < row; i++){

        for(int j = 0; j < col; j++){
            gsl_matrix_set(input_weight, i, j, gsl_ran_gaussian(r, sigma));
        }
    }
    gsl_rng_free(r);
    return input_weight;
}

double get_output_error(gsl_matrix* target, gsl_matrix* output, gsl_matrix* output_error)
{
	double error = 0.0;
	for (int i = 0; i < OUTPUT_NODE; i++){
		gsl_matrix_set(output_error, i, 0, (gsl_matrix_get(target, i, 0) - gsl_matrix_get(output, i, 0)));
	}

	for (int i = 0; i < OUTPUT_NODE; i++){
		error += pow(gsl_matrix_get(output_error, i, 0), 2);
	}

	return error;
}

void training(gsl_matrix* better_input, gsl_matrix* target, gsl_matrix* input_weight, gsl_matrix* hidden_weight, gsl_matrix* hidden_x, gsl_matrix* hidden_o, gsl_matrix* output_o, gsl_matrix* output_x, gsl_matrix* output_error, gsl_matrix* hidden_error, double* error)
{
	gsl_matrix* delta_hidden_weight = gsl_matrix_alloc(OUTPUT_NODE, HIDDEN_NODE);
	gsl_matrix* delta_input_weight = gsl_matrix_alloc(HIDDEN_NODE, DATA_NUM_WITHOUT_LABEL);
	gsl_matrix* temp_hidden = gsl_matrix_alloc(OUTPUT_NODE, 1);
	gsl_matrix* temp_input = gsl_matrix_alloc(HIDDEN_NODE, 1);

	 gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, input_weight, better_input, 0.0, hidden_x);  // X = W*I
	 for(int i = 0; i < HIDDEN_NODE; i++){
	 	gsl_matrix_set(hidden_o, i, 0, SIGMOID(gsl_matrix_get(hidden_x, i, 0)));
	 }

	 gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, hidden_weight, hidden_o, 0.0, output_x);
	 for(int i = 0; i < OUTPUT_NODE; i++){
	 	gsl_matrix_set(output_o, i, 0, SIGMOID(gsl_matrix_get(output_x, i, 0)));
	 }

	 *error = get_output_error(target, output_o, output_error);
	 gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, hidden_weight, output_error, 0.0, hidden_error); // Error hidden

	 for(int i = 0; i < OUTPUT_NODE; i++){
	 	gsl_matrix_set(temp_hidden, i, 0, (LEARN_RATE * gsl_matrix_get(output_error, i, 0) * gsl_matrix_get(output_o, i, 0) * (1.0 - gsl_matrix_get(output_o, i, 0))));
	 }
	 gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, temp_hidden, hidden_o, 0.0, delta_hidden_weight);

	 for(int i = 0; i < HIDDEN_NODE; i++){
	 	gsl_matrix_set(temp_input, i, 0, (LEARN_RATE * gsl_matrix_get(hidden_error, i, 0) * gsl_matrix_get(hidden_o, i, 0) * (1.0 - gsl_matrix_get(hidden_error, i, 0))));
	 }
	 gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, temp_input, better_input, 0.0, delta_input_weight);

	 gsl_matrix_add(hidden_weight, delta_hidden_weight);
	 gsl_matrix_add(input_weight, delta_input_weight);


	 gsl_matrix_free(delta_input_weight);
	 gsl_matrix_free(delta_hidden_weight);
	 gsl_matrix_free(temp_input);
	 gsl_matrix_free(temp_hidden);

}

void store_weight(gsl_matrix* input_weight, gsl_matrix* hidden_weight)
{
	FILE* fp = fopen("Good_Weight", "w+");

	for(int i = 0; i < HIDDEN_NODE; i++){
		for(int j = 0; j < DATA_NUM_WITHOUT_LABEL; j++){

			fprintf(fp, "%f,", gsl_matrix_get(input_weight, i, j));
		}
	}

	fseek(fp, -1, SEEK_CUR);
	fputc('\n', fp);

	for(int i = 0; i < OUTPUT_NODE; i++){
		for(int j = 0; j < HIDDEN_NODE; j++){

			fprintf(fp, "%f,", gsl_matrix_get(hidden_weight, i, j));
		}
	}
	fseek(fp, -1, SEEK_CUR);
	fputc('\n', fp);

	fclose(fp);
}

void load_weight(gsl_matrix* input_weight, gsl_matrix* hidden_weight)
{
	FILE* fp = fopen("Good_Weight", "r");
	char num_temp[20];
	double array[HIDDEN_NODE*DATA_NUM_WITHOUT_LABEL];
	char c;
	int j = 0, k;

	if(fp == NULL){
		fprintf(stderr, "No Good_Weight here, Please training first\n");
		exit(EXIT_FAILURE);
	}

	while((c = fgetc(fp)) != '\n'){

		if(c == ','){
			k = 0;
			array[j++] = atof(num_temp);
			bzero(num_temp, sizeof(num_temp));
			continue;
		}
		num_temp[k++] = c;
	}

	array[j] = atof(num_temp);

	j = 0;

	for(int n = 0; n < HIDDEN_NODE; n++){
		for(int m = 0; m < DATA_NUM_WITHOUT_LABEL; m++){
			gsl_matrix_set(input_weight, n, m, array[j++]);
		}
	}

	j = 0;
	k = 0;
	while((c = fgetc(fp)) != '\n'){

		if(c == ','){
			k = 0;
			array[j++] = atof(num_temp);
			bzero(num_temp, sizeof(num_temp));
			continue;
		}
		num_temp[k++] = c;
	}

	array[j] = atof(num_temp);

	j = 0;

	for(int n = 0; n < OUTPUT_NODE; n++){
		for(int m = 0; m < HIDDEN_NODE; m++){
			gsl_matrix_set(hidden_weight, n, m, array[j++]);
		}
	}

	fclose(fp);
}

int recognize(gsl_matrix* better_input, gsl_matrix* input_weight, gsl_matrix* hidden_weight, gsl_matrix* hidden_x, gsl_matrix* hidden_o, gsl_matrix* output_x, gsl_matrix* output_o)
{
	size_t row, col;
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, input_weight, better_input, 0.0, hidden_x);  // X = W*I
	 for(int i = 0; i < HIDDEN_NODE; i++){
	 	gsl_matrix_set(hidden_o, i, 0, SIGMOID(gsl_matrix_get(hidden_x, i, 0)));
	 }

	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, hidden_weight, hidden_o, 0.0, output_x);
	 for(int i = 0; i < OUTPUT_NODE; i++){
	 	gsl_matrix_set(output_o, i, 0, SIGMOID(gsl_matrix_get(output_x, i, 0)));
	 }

	 gsl_matrix_max_index(output_o, &row, &col);

	 return (int)row;
}

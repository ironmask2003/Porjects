/*
 * k-Means clustering algorithm
 *
 * CUDA version
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.0
 *
 * (c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include <cuda.h>


#define MAXLINE 2000
#define MAXCAD 200

//Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/*
 * Macros to show errors when calling a CUDA library function,
 * or after launching a kernel
 */
#define CHECK_CUDA_CALL( a )	{ \
	cudaError_t ok = a; \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}
#define CHECK_CUDA_LAST()	{ \
	cudaError_t ok = cudaGetLastError(); \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA last in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}

/* 
Function showFileError: It displays the corresponding error during file reading.
*/
void showFileError(int error, char* filename)
{
	printf("Error\n");
	switch (error)
	{
		case -1:
			fprintf(stderr,"\tFile %s has too many columns.\n", filename);
			fprintf(stderr,"\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n", MAXLINE);
			break;
		case -2:
			fprintf(stderr,"Error reading file: %s.\n", filename);
			break;
		case -3:
			fprintf(stderr,"Error writing file: %s.\n", filename);
			break;
	}
	fflush(stderr);	
}

/* 
Function readInput: It reads the file to determine the number of rows and columns.
*/
int readInput(char* filename, int *lines, int *samples)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines, contsamples = 0;
    
    contlines = 0;

    if ((fp=fopen(filename,"r"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL) 
		{
			if (strchr(line, '\n') == NULL)
			{
				return -1;
			}
            contlines++;       
            ptr = strtok(line, delim);
            contsamples = 0;
            while(ptr != NULL)
            {
            	contsamples++;
				ptr = strtok(NULL, delim);
	    	}	    
        }
        fclose(fp);
        *lines = contlines;
        *samples = contsamples;  
        return 0;
    }
    else
	{
    	return -2;
	}
}

/* 
Function readInput2: It loads data from file.
*/
int readInput2(char* filename, float* data)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;
    
    if ((fp=fopen(filename,"rt"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL)
        {         
            ptr = strtok(line, delim);
            while(ptr != NULL)
            {
            	data[i] = atof(ptr);
            	i++;
				ptr = strtok(NULL, delim);
	   		}
	    }
        fclose(fp);
        return 0;
    }
    else
	{
    	return -2; //No file found
	}
}

/* 
Function writeResult: It writes in the output file the cluster of each sample (point).
*/
int writeResult(int *classMap, int lines, const char* filename)
{	
    FILE *fp;
    
    if ((fp=fopen(filename,"wt"))!=NULL)
    {
        for(int i=0; i<lines; i++)
        {
        	fprintf(fp,"%d\n",classMap[i]);
        }
        fclose(fp);  
   
        return 0;
    }
    else
	{
    	return -3; //No file found
	}
}

/*

Function initCentroids: This function copies the values of the initial centroids, using their 
position in the input data structure as a reference map.
*/
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K)
{
	int i;
	int idx;
	for(i=0; i<K; i++)
	{
		idx = centroidPos[i];
		memcpy(&centroids[i*samples], &data[idx*samples], (samples*sizeof(float)));
	}
}

/*
Function euclideanDistance: Euclidean distance
This function could be modified
*/
__device__ float euclideanDistance(float *point, float *center, int samples)
{
	float dist=0.0;
	for(int i=0; i<samples; i++) 
	{
		dist+= (point[i]-center[i])*(point[i]-center[i]);
	}
	dist = sqrt(dist);
	return(dist);
}

/*
Function zeroFloatMatriz: Set matrix elements to 0
This function could be modified
*/
void zeroFloatMatriz(float *matrix, int rows, int columns)
{
	int i,j;
	for (i=0; i<rows; i++)
		for (j=0; j<columns; j++)
			matrix[i*columns+j] = 0.0;	
}

/*
Function zeroIntArray: Set array elements to 0
This function could be modified
*/
void zeroIntArray(int *array, int size)
{
	int i;
	for (i=0; i<size; i++)
		array[i] = 0;	
}


// Variabili global
__constant__ int gpu_samples;
__constant__ int gpu_K;
__constant__ int gpu_lines;

// Funzione che assegna ad ogni punto il centroide più vicino
__global__ void assign_centroids(float *d_data, float *d_centroids, int *d_classMap, int* changes, int* class_var)
{
	int thread_index = (blockIdx.y * gridDim.x * blockDim.x * blockDim.y) + (blockIdx.x * blockDim.x * blockDim.y) +
							(threadIdx.y * blockDim.x) +
							threadIdx.x;

	if(thread_index < gpu_lines)
	{
		class_var=1;
		float dist, minDist=FLT_MAX;

		for(int j=0; j<gpu_K; j++)
		{
			dist=euclideanDistance(&d_data[thread_index*gpu_samples], &d_centroids[j*gpu_samples], gpu_samples);

			if(dist < minDist)
			{
				minDist=dist;
				class_var=j+1;
			}
		}

		if(d_classMap[thread_index]!=class_var)
		{
			atomicAdd(changes, 1);
		}

		d_classMap[thread_index]=class_var;
	}
}

int main(int argc, char* argv[])
{

	//START CLOCK***************************************
	double start, end;
	start = omp_get_wtime();
	//**************************************************
	/*
	* PARAMETERS
	*
	* argv[1]: Input data file
	* argv[2]: Number of clusters
	* argv[3]: Maximum number of iterations of the method. Algorithm termination condition.
	* argv[4]: Minimum percentage of class changes. Algorithm termination condition.
	*          If between one iteration and the next, the percentage of class changes is less than
	*          this percentage, the algorithm stops.
	* argv[5]: Precision in the centroid distance after the update.
	*          It is an algorithm termination condition. If between one iteration of the algorithm 
	*          and the next, the maximum distance between centroids is less than this precision, the
	*          algorithm stops.
	* argv[6]: Output file. Class assigned to each point of the input file.
	* */
	if(argc !=  7)
	{
		fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		exit(-1);
	}

	// Reading the input data
	// lines = number of points; samples = number of dimensions per point
	int lines = 0, samples= 0;  
	
	int error = readInput(argv[1], &lines, &samples);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}
	
	float *data = (float*)calloc(lines*samples,sizeof(float));
	if (data == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}
	error = readInput2(argv[1], data);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}

	// Parameters
	int K=atoi(argv[2]); 
	int maxIterations=atoi(argv[3]);
	int minChanges= (int)(lines*atof(argv[4])/100.0);
	float maxThreshold=atof(argv[5]);

	int *centroidPos = (int*)calloc(K,sizeof(int));
	float *centroids = (float*)calloc(K*samples,sizeof(float));
	int *classMap = (int*)calloc(lines,sizeof(int));

    if (centroidPos == NULL || centroids == NULL || classMap == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

	// Initial centrodis
	srand(0);
	int i;
	for(i=0; i<K; i++) 
		centroidPos[i]=rand()%lines;
	
	// Loading the array of initial centroids with the data from the array data
	// The centroids are points stored in the data array.
	initCentroids(data, centroids, centroidPos, samples, K);


	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);
	
	//END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\nMemory allocation: %f seconds\n", end - start);
	fflush(stdout);

	CHECK_CUDA_CALL( cudaSetDevice(0) );
	CHECK_CUDA_CALL( cudaDeviceSynchronize() );
	//**************************************************
	//START CLOCK***************************************
	start = omp_get_wtime();
	//**************************************************
	char *outputMsg = (char *)calloc(10000,sizeof(char));
	char line[100];

	int j;
	int class_var;
	float dist, minDist;
	int it=0;
	int changes = 0;
	float maxDist;

	//pointPerClass: number of points classified in each class
	//auxCentroids: mean of the points in each class
	int *pointsPerClass = (int *)malloc(K*sizeof(int));
	float *auxCentroids = (float*)malloc(K*samples*sizeof(float));
	float *distCentroids = (float*)malloc(K*sizeof(float)); 
	if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

	// Inizializzazione delle costanti per le gpu
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_K, &K, sizeof(int)));
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_samples, &samples, sizeof(int)));
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(gpu_lines, &lines, sizeof(int)));

	// Adapt to the number of points
	int pts_grid_size = samples / (32 * 32) + 1;
	int K_grid_size = K / (32 * 32) + 1;

	// Set carveout to be of maximum size available
	int carveout = cudaSharedmemCarveoutMaxShared;

	CHECK_CUDA_CALL(cudaFuncSetAttribute(assign_centroids, cudaFuncAttributePreferredSharedMemoryCarveout, carveout));

	dim3 gen_block(32, 32);
	dim3 dyn_grid_pts(pts_grid_size);
	dim3 dyn_grid_cent(K_grid_size);

	// Copy data to device
	float *d_data;
	int *d_classMap;
	float *d_centroids;
	int *d_pointsPerClass;
	float *d_auxCentroids;
	float *d_distCentroids;

	CHECK_CUDA_CALL( cudaMalloc(&d_data, lines*samples*sizeof(float)) );
	CHECK_CUDA_CALL( cudaMemcpy(d_data, data, lines*samples*sizeof(float), cudaMemcpyHostToDevice) );

	CHECK_CUDA_CALL( cudaMalloc(&d_classMap, lines*sizeof(int)) );
	CHECK_CUDA_CALL( cudaMemcpy(d_classMap, classMap, lines*sizeof(int), cudaMemcpyHostToDevice) );
	
	CHECK_CUDA_CALL( cudaMalloc(&d_centroids, K*samples*sizeof(float)) );
	CHECK_CUDA_CALL( cudaMemcpy(d_centroids, centroids, K*samples*sizeof(float), cudaMemcpyHostToDevice) );
	
	CHECK_CUDA_CALL( cudaMalloc(&d_pointsPerClass, K*sizeof(int)) );
	CHECK_CUDA_CALL( cudaMemcpy(d_pointsPerClass, pointsPerClass, K*sizeof(int), cudaMemcpyHostToDevice) );
	
	CHECK_CUDA_CALL( cudaMalloc(&d_auxCentroids, K*samples*sizeof(float)) );
	CHECK_CUDA_CALL( cudaMemcpy(d_auxCentroids, auxCentroids, K*samples*sizeof(float), cudaMemcpyHostToDevice) );
	
	CHECK_CUDA_CALL( cudaMalloc(&d_distCentroids, K*sizeof(float)) );
	CHECK_CUDA_CALL( cudaMemcpy(d_distCentroids, distCentroids, K*sizeof(float), cudaMemcpyHostToDevice) );

	do{
		it++;
	
		//1. Calculate the distance from each point to the centroid
		//Assign each point to the nearest centroid.

		CHECK_CUDA_CALL(cudaMemset(changes, 0, sizeof(int)));

		// Synschronize
		CHECK_CUDA_CALL(cudaDeviceSynchronize());

		assign_centroids<<<dyn_grid_pts, gen_block, K * lines * sizeof(float)>>>(d_data, d_centroids, d_classMap, changes, class_var);

		// Syncronize
		CHECK_CUDA_CALL(cudaDeviceSynchronize());

		CHECK_CUDA_CALL( cudaMemset(d_pointsPerClass, 0, K*sizeof(int)) );
		CHECK_CUDA_CALL( cudaMemset(d_auxCentroids, 0, K*samples*sizeof(float)) );

		for(i=0; i<lines; i++) 
		{
			class_var=classMap[i];
			pointsPerClass[class_var-1] = pointsPerClass[class_var-1] +1;
			for(j=0; j<samples; j++){
				auxCentroids[(class_var-1)*samples+j] += data[i*samples+j];
			}
		}

		for(i=0; i<K; i++) 
		{
			for(j=0; j<samples; j++){
				auxCentroids[i*samples+j] /= pointsPerClass[i];
			}
		}
		
		maxDist=FLT_MIN;
		for(i=0; i<K; i++){
			distCentroids[i]=euclideanDistance(&centroids[i*samples], &auxCentroids[i*samples], samples);
			if(distCentroids[i]>maxDist) {
				maxDist=distCentroids[i];
			}
		}
		memcpy(centroids, auxCentroids, (K*samples*sizeof(float)));
		
		sprintf(line,"\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
		outputMsg = strcat(outputMsg,line);

	} while((changes>minChanges) && (it<maxIterations) && (maxDist>maxThreshold));

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */
	// Output and termination conditions
	printf("%s",outputMsg);	

	CHECK_CUDA_CALL( cudaDeviceSynchronize() );

	//END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\nComputation: %f seconds", end - start);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = omp_get_wtime();
	//**************************************************

	

	if (changes <= minChanges) {
		printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
	}
	else if (it >= maxIterations) {
		printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
	}
	else {
		printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
	}	

	// Writing the classification of each point to the output file.
	error = writeResult(classMap, lines, argv[6]);
	if(error != 0)
	{
		showFileError(error, argv[6]);
		exit(error);
	}

	//Free memory
	free(data);
	free(classMap);
	free(centroidPos);
	free(centroids);
	free(distCentroids);
	free(pointsPerClass);
	free(auxCentroids);

	//END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\n\nMemory deallocation: %f seconds\n", end - start);
	fflush(stdout);
	//***************************************************/
	return 0;
}

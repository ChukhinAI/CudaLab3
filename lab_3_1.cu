#include <iostream>

using namespace std;

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

//calculate<<<(rowsY + 255) / 256, 256>>>(dev_original_matrix, dev_result, elementsQuantity, rowsY, colsX, border_number); <<<кол-во блоков в grid, кол-во потоков>>>
// + 255 для того, чтобы точно уместить все данные
// размер массива ограничен максимальным размером пространства потоков = 256
  __global__ void calculate(int *dev_original_matrix, int *dev_result, int elementsQuantity, int rowsY, int colsX, int border_number)
  {
    int current_row_number = threadIdx.x + blockIdx.x * blockDim.x; // номер строки в изображении
	// Grid -> блок -> поток(поток в блоке, блок в сетке), один поток запускает функцию calculate один раз
	// blockIdx - номер блока в 1D-grid, blockDim - кол-во блоков в одном потоке

	int r;
	int count_points = 0;
	for (int i = 1; i < colsX - 1; i++) // крайние не считаются
    {
        //r = matrix[i * colsX + j - 1] - matrix[i * colsX + j + 1]; // мб надо делить на 2, но в Гонсалесе вот так(см градиент Собела/Собеля);
		
		r = dev_original_matrix[current_row_number * colsX + i - 1] - dev_original_matrix[current_row_number * colsX + i + 1];
		if (r > border_number)
		{
			count_points++;
		}
    }
	dev_result[current_row_number] = count_points;
	
	
	//cout << "dev_result in GPU claculate :" << "\n";
	/*
	for (int i = 0; i < current_row_number; i++)
	{
		cout << dev_result[i] << " ";
	}
	cout << '\n';
	*/
}

//СОХРАНЯТЬ РЕЗУЛЬТАТ В ЛОК ПЕРЕМ ПОТОМ В РЕЗ МАТРИЦУ
__global__ void goodCalculation(int *dev_original_matrix, int *dev_result, int elementsQuantity,
									int rowsY, int colsX, int border_number)
{
	
	int rowsCountBeforeOrangeLine = blockIdx.x * blockDim.x;
    //int bigRowNumber = blockIdx.x * blockDim.x + threadIdx.x;

    int cacheWidth = 32;	 // original
    int rectangleHeight = 8; // original

    //int rectangleInRowQuantity = colsX / cacheWidth; // original
	int rectangleInRowQuantity = (colsX - 2) / (cacheWidth - 2);

    __shared__ int cache[256][33]; 

	int r;
	int count_points = 0;

	int rowInCache = threadIdx.x / cacheWidth;  // номер строки в верхнем ЗП (первый элемент)
	int currentRowInCache = rowInCache;
	int columnInCache = threadIdx.x % cacheWidth;
	int pixelCountUpperRowInTopGreenRect = (rowsCountBeforeOrangeLine + rowInCache) * colsX;	
	int indexTopPixelInCurrentFPInsideImage = pixelCountUpperRowInTopGreenRect + columnInCache;
	int verticalStep = rectangleHeight * colsX;	
    for (int stringIteration = 0; stringIteration < rectangleInRowQuantity; stringIteration++)
    {		
		int currentPixelInImage = indexTopPixelInCurrentFPInsideImage;
      for (int levelInCache = 0; levelInCache < cacheWidth; levelInCache++)
      {		
	    cache[currentRowInCache][columnInCache] = dev_original_matrix[currentPixelInImage]; 										 
        currentRowInCache += rectangleHeight; 
		currentPixelInImage += verticalStep; // verticalStep по ЗП вниз
       
      }
	  indexTopPixelInCurrentFPInsideImage += 30; // переход к след ФП
	  currentRowInCache = rowInCache;
      __syncthreads();
	  
	  r = 0;
	  
 // тут начинаются ошибки с неправильным обращенем к памяти - fixed
	  for (int i = 1; i < cacheWidth - 1; i++)
      {
		r = cache[threadIdx.x][i - 1] - cache[threadIdx.x][i + 1];
		if (r > border_number) // ошибка
			count_points = count_points + 1;		
      }

      __syncthreads();
    }

	dev_result[rowsCountBeforeOrangeLine + threadIdx.x] = count_points; // ошибка с неправильным обращенем к памяти - fixed
}

void printMatrix(int* matrix, int colsX, int rowsY)
{
  for (int i = 0; i < rowsY; i++)
  {
    for (int j = 0; j < colsX; j++)
    {
        cout << matrix[i * colsX + j] << "\t";
    }
    cout << "\n";
  }
}

bool checkResult(int* host_result, int* result, int colsX, int rowsY)
{

  for (int i = 0; i < 20; i++)
  {
	cout << "host_result[ " << i << " ] = " << host_result[i] << '\n';
	
  }
  
    for (int i = 0; i < 20; i++)
  {

	cout << "result[ " << i << " ] = " << result[i] << '\n';
  }
  
  for (int i = 0; i < rowsY; i++)
  {
    if (host_result[i] != result[i])
    {
	//cout << "host_result[ " << i << " ] = " << host_result[i] << '\n';
	//cout << "result[ " << i << " ] = " << result[i] << '\n';
      return false;
    }
  }

  return true;
}

int main(void)
{
    cudaEvent_t startCUDA, stopCUDA, startOptimalCUDA, stopOptimalCUDA;
    clock_t startCPU;
    float elapsedUsualTimeCUDA, elapsedTimeCPU, elapsedOptimalTime;

    // 13. Создайте детектор вертикальных границ на изображении (в градациях серого). 
	// Функция должна для каждой строки считать количество точек, в которых производная цвета по горизонтали больше заданного значения.
	// Все изображения хранятся в памяти по строкам.

    int colsX = 1502; 		//  пикселей 30 * 50 + 2 = 1502
    int rowsY = 17920; 		//  пикселей 256 * 70 = 17920
    int elementsQuantity = colsX * rowsY;
    cout << "Size in Mbs = " << elementsQuantity * sizeof(int) / 1048576.0 << "\n";
    int *matrix = new int[elementsQuantity];

    for (int i = 0; i < rowsY; i++)
    {
      for (int j = 0; j < colsX; j++)
      {
          matrix[i * colsX + j] = rand() % 255; // filling matrix
		  //matrix[i * colsX + j] = (i * colsX + j) * 10 * i;
      }
    }

    //printMatrix(matrix, colsX, rowsY);

    int border_number = 10; // -410
	cout << "border_number = " << border_number << '\n';

    startCPU = clock();
    int *result = new int[rowsY];
    //int *count_points = new int[rowsY];
	int r;
	int count_points;
    for (int i = 0; i < rowsY; i++) // alg CPU func 
    {
      //int r = 0;
	  //int count_points = 0;
	  count_points = 0;

      for (int j = 1; j < colsX - 1; j++)
      {
       //r = r + matrix[i * colsX + j]; // original
       r = matrix[i * colsX + j - 1] - matrix[i * colsX + j + 1]; // мб надо делить на 2, но в Гонсалесе вот так(см градиент Собела/Собеля);
	   //dI = dy/dx -> у нас только вертикальные границы, поэтому считаем приращение только по x
	   //cout << "r = " << r << "\n";
        if (r > border_number)
        {
          //cout << "r = " << r << "\n";
		  //cout << "found one" << "\n";
          count_points++;
        }
		
      }
      result[i] = count_points;
	  
	  //cout << "in " << i << " row found " << result[i] << " points" << "\n";
    }
  
  /*
	cout << "result in CPU :" << "\n";
	for (int i = 0; i < rowsY; i++)
	{
		cout << result[i] << " ";
	}
	cout << '\n';
*/

    clock_t end = clock();
    elapsedTimeCPU = (double)(end-startCPU)/CLOCKS_PER_SEC;
    cout << "CPU calculating time = " << elapsedTimeCPU * 1000 << " ms\n";
    cout << "CPU memory throughput = " << elementsQuantity *sizeof(int)/elapsedTimeCPU/1024/1024/1024 << " Gb/s\n";
    
    cout << "\n";

    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    int *dev_original_matrix, *dev_result;
    int *host_original_matrix, * host_result;

    host_original_matrix = matrix;
    host_result = new int[rowsY];
    for (int i = 0; i < rowsY; i++)
    {
      host_result[i] = 0;
    }

    CHECK( cudaMalloc(&dev_original_matrix, elementsQuantity * sizeof(int)));
    CHECK( cudaMalloc(&dev_result, rowsY * sizeof(int)));

    CHECK( cudaMemcpy(dev_original_matrix, host_original_matrix, elementsQuantity * sizeof(int), cudaMemcpyHostToDevice));
    CHECK( cudaMemcpy(dev_result, host_result, rowsY * sizeof(int), cudaMemcpyHostToDevice));
	
    cudaEventRecord(startCUDA, 0);
    calculate<<<(rowsY + 255) / 256, 256>>>(dev_original_matrix, dev_result, elementsQuantity, rowsY, colsX, border_number);
    cudaEventRecord(stopCUDA, 0);
    cout << "FINISH" << '\n';

    cudaEventSynchronize(stopCUDA);
    CHECK(cudaGetLastError());

    cudaEventElapsedTime(&elapsedUsualTimeCUDA, startCUDA, stopCUDA);

    cout << "CUDA sum time = " << elapsedUsualTimeCUDA << " ms\n";
    cout << "CUDA memory throughput = " << elementsQuantity * sizeof(int) / elapsedUsualTimeCUDA/1024/1024/1.024 << " Gb/s\n";
    CHECK( cudaMemcpy(host_result, dev_result, rowsY * sizeof(int),cudaMemcpyDeviceToHost));


/*
	cout << '\n' << "host_result = " << '\n';
	printMatrix(host_result, 1, rowsY);

	cout << '\n' << "result = " << '\n';
	printMatrix(result, 1, rowsY);
*/	


    cout << "result was correct " << checkResult(host_result, result, colsX, rowsY) << "\n";
    cout << "Data size = " << (float)4 * elementsQuantity / 1024 / 1024 << "\n";

    CHECK( cudaFree(dev_original_matrix));
    CHECK( cudaFree(dev_result));

//}
///*
    //**********************************************************************************************
    //ХОРОШЕЕ УМНОЖЕНИЕ

    cudaEventCreate(&startOptimalCUDA);
    cudaEventCreate(&stopOptimalCUDA);

    int* good_host_result = new int[rowsY];
    for (int i = 0; i < rowsY; i++)
    {
      good_host_result[i] = 0; // 0
    }

    int *good_dev_result;
    CHECK( cudaMalloc(&dev_original_matrix, elementsQuantity * sizeof(int)));
    CHECK( cudaMalloc(&good_dev_result,rowsY * sizeof(int)));

    CHECK( cudaMemcpy(dev_original_matrix, host_original_matrix, elementsQuantity * sizeof(int), cudaMemcpyHostToDevice));
    CHECK( cudaMemcpy(good_dev_result, good_host_result, rowsY * sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(startOptimalCUDA, 0);
    goodCalculation<<<(rowsY + 255) / 256, 256>>>(dev_original_matrix, good_dev_result, elementsQuantity, rowsY, colsX, border_number);


	//cout << '\n' << "good_host_result = " << '\n'; 
	//printMatrix(good_host_result, 1, rowsY);

	//cout << '\n' << "good_dev_result = " << '\n'; // good_dev_result пустая?
	//printMatrix(good_dev_result, 1, rowsY);


    cudaEventRecord(stopOptimalCUDA, 0);
    CHECK( cudaMemcpy(good_host_result, good_dev_result, rowsY * sizeof(int),cudaMemcpyDeviceToHost));
    cout << ("OPTIMAL SUMMATION WAS FINISHED");
	


    cudaEventElapsedTime(&elapsedOptimalTime, startOptimalCUDA, stopOptimalCUDA);

    cout << "CUDA GOOD (OPTIMAL) sum time = " << elapsedOptimalTime << " ms\n";
    cout << "CUDA GOOD (OPTIMAL) memory throughput = " << elementsQuantity * sizeof(int) / elapsedOptimalTime/1024/1024/1.024 << " Gb/s\n";




	//cout << '\n' << "good_host_result = " << '\n'; 
	//printMatrix(good_host_result, 1, rowsY);


    cout << "result was correct" <<  checkResult(good_host_result, result, colsX, rowsY) << "\n";
    cout << "Data size = " << (float)4 * elementsQuantity / 1024 / 1024 << "\n"; // float original, ok

    CHECK( cudaFree(dev_original_matrix));
    CHECK( cudaFree(good_dev_result));
    return 0;
}
//*/

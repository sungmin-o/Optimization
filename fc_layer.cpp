#include <stdio.h>
#include <immintrin.h>

void fc_layer(size_t data_cnt, size_t input_dim, size_t output_dim, float* matrix, float* bias, float* input, float* output) {
    // loop over input instances //input_dim == 1
	size_t bsize = 512;
	size_t n1, n2;
	__m256 result, result1, result2, result3, MUL, MUL1, MUL2, MUL3, mul, mul1, mul2, mul3, vvec, vvec1;
	float* ptr; float* ptr1; float* ptr2; float* ptr3;

    for(size_t ii = 0; ii < data_cnt; ii += bsize){
		for(size_t kk = 0; kk < input_dim; kk += bsize){
			for(size_t jj = 0; jj < output_dim; jj += bsize){
				for(size_t i = ii; i < ii + bsize; i+=4){
					n1 = i * input_dim;
					n2 = kk * output_dim;

					ptr = output + i * output_dim;
					ptr1 = output + (i + 1) * output_dim;
					ptr2 = output + (i + 2) * output_dim;
					ptr3 = output + (i + 3) * output_dim;

					for(size_t k = kk; k < kk + bsize; k+=2){

						__m256 vec[4] = {_mm256_set1_ps(input[n1 + k]), _mm256_set1_ps(input[n1 + input_dim + k]), _mm256_set1_ps(input[n1 + input_dim * 2 + k]), _mm256_set1_ps(input[n1 + input_dim * 3 + k])};
						__m256 vec1[4] = {_mm256_set1_ps(input[n1 + k + 1]), _mm256_set1_ps(input[n1 + input_dim + 1 + k ]), _mm256_set1_ps(input[n1 + input_dim * 2 + 1 + k]), _mm256_set1_ps(input[n1 + input_dim * 3 + 1 + k])};

						for(size_t j = jj; j < jj + bsize; j += 32){
							MUL = _mm256_load_ps(matrix + n2 + j);
							MUL1 = _mm256_load_ps(matrix + n2 + j + 8);
							MUL2 = _mm256_load_ps(matrix + n2 + j + 16);
							MUL3 = _mm256_load_ps(matrix + n2 + j + 24);

							mul = _mm256_load_ps(matrix + n2 + output_dim + j);
							mul1 = _mm256_load_ps(matrix + n2 + output_dim + j + 8);
							mul2 = _mm256_load_ps(matrix + n2 + output_dim + j + 16);
							mul3 = _mm256_load_ps(matrix + n2 + output_dim + j + 24);

							//****************************************************//
							vvec = vec[0];
							vvec1 = vec1[0];

							result = _mm256_fmadd_ps(vvec1, mul,_mm256_mul_ps(vvec, MUL));
							_mm256_store_ps(ptr + j, _mm256_add_ps(result, _mm256_load_ps(ptr + j)));

							result1 = _mm256_fmadd_ps(vvec1, mul1,_mm256_mul_ps(vvec, MUL1));
							_mm256_store_ps(ptr + j + 8, _mm256_add_ps(result1, _mm256_load_ps(ptr + j + 8)));

							result2 = _mm256_fmadd_ps(vvec1, mul2,_mm256_mul_ps(vvec, MUL2));
							_mm256_store_ps(ptr + j + 16, _mm256_add_ps(result2, _mm256_load_ps(ptr + j + 16)));

							result3 = _mm256_fmadd_ps(vvec1, mul3,_mm256_mul_ps(vvec, MUL3));
							_mm256_store_ps(ptr + j + 24, _mm256_add_ps(result3, _mm256_load_ps(ptr + j + 24)));

							//****************************************************//

							vvec = vec[1];
							vvec1 = vec1[1];
							
							result = _mm256_fmadd_ps(vvec1, mul,_mm256_mul_ps(vvec, MUL));
							_mm256_store_ps(ptr1 + j, _mm256_add_ps(result, _mm256_load_ps(ptr1 + j)));

							result1 = _mm256_fmadd_ps(vvec1, mul1,_mm256_mul_ps(vvec, MUL1));
							_mm256_store_ps(ptr1 + j + 8, _mm256_add_ps(result1, _mm256_load_ps(ptr1 + j + 8)));

							result2 = _mm256_fmadd_ps(vvec1, mul2,_mm256_mul_ps(vvec, MUL2));
							_mm256_store_ps(ptr1 + j + 16, _mm256_add_ps(result2, _mm256_load_ps(ptr1 + j + 16)));

							result3 = _mm256_fmadd_ps(vvec1, mul3,_mm256_mul_ps(vvec, MUL3));
							_mm256_store_ps(ptr1 + j + 24, _mm256_add_ps(result3, _mm256_load_ps(ptr1 + j + 24)));

							//****************************************************//

							vvec = vec[2];
							vvec1 = vec1[2];
							
							result = _mm256_fmadd_ps(vvec1, mul,_mm256_mul_ps(vvec, MUL));
							_mm256_store_ps(ptr2 + j, _mm256_add_ps(result, _mm256_load_ps(ptr2 + j)));

							result1 = _mm256_fmadd_ps(vvec1, mul1,_mm256_mul_ps(vvec, MUL1));
							_mm256_store_ps(ptr2 + j + 8, _mm256_add_ps(result1, _mm256_load_ps(ptr2 + j + 8)));

							result2 = _mm256_fmadd_ps(vvec1, mul2,_mm256_mul_ps(vvec, MUL2));
							_mm256_store_ps(ptr2 + j + 16, _mm256_add_ps(result2, _mm256_load_ps(ptr2 + j + 16)));

							result3 = _mm256_fmadd_ps(vvec1, mul3,_mm256_mul_ps(vvec, MUL3));
							_mm256_store_ps(ptr2 + j + 24, _mm256_add_ps(result3, _mm256_load_ps(ptr2 + j + 24)));

							//****************************************************//

							vvec = vec[3];
							vvec1 = vec1[3];

							result = _mm256_fmadd_ps(vvec1, mul,_mm256_mul_ps(vvec, MUL));
							_mm256_store_ps(ptr3 + j, _mm256_add_ps(result, _mm256_load_ps(ptr3 + j)));

							result1 = _mm256_fmadd_ps(vvec1, mul1,_mm256_mul_ps(vvec, MUL1));
							_mm256_store_ps(ptr3 + j + 8, _mm256_add_ps(result1, _mm256_load_ps(ptr3 + j + 8)));

							result2 = _mm256_fmadd_ps(vvec1, mul2,_mm256_mul_ps(vvec, MUL2));
							_mm256_store_ps(ptr3 + j + 16, _mm256_add_ps(result2, _mm256_load_ps(ptr3 + j + 16)));

							result3 = _mm256_fmadd_ps(vvec1, mul3,_mm256_mul_ps(vvec, MUL3));
							_mm256_store_ps(ptr3 + j + 24, _mm256_add_ps(result3, _mm256_load_ps(ptr3 + j + 24)));

							//****************************************************//
						}
						n2 += output_dim * 2;
					}
				}
			}
		}
	}

	size_t nn = 0;
	__m256 bias_vec0, bias_vec1, output_vec0, output_vec1, zero;
	zero = _mm256_setzero_ps();

	for(size_t i = 0; i < data_cnt; i++){
		for(size_t j = 0; j < output_dim; j += 16){
			bias_vec0 = _mm256_load_ps(bias + j);
			bias_vec1 = _mm256_load_ps(bias + j + 8);

			output_vec0 = _mm256_load_ps(output + nn + j);
			output_vec1 = _mm256_load_ps(output + nn + j + 8);

			output_vec0 = _mm256_add_ps(output_vec0, bias_vec0);
			output_vec1 = _mm256_add_ps(output_vec1, bias_vec1);

			output_vec0 = _mm256_max_ps(output_vec0, zero);
			output_vec1 = _mm256_max_ps(output_vec1, zero);

			_mm256_store_ps(output + nn + j, output_vec0);
			_mm256_store_ps(output + nn + j + 8, output_vec1);
		}
		nn += output_dim;
	}
	
}
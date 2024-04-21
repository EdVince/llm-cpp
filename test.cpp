#include "armpl.h"
#include <stdio.h>

int main()
{
  int i=0;
  __fp16 A[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};
  __fp16 B[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};
  __fp16 C[9] = {.5,.5,.5,.5,.5,.5,.5,.5,.5};
  cblas_hgemm(CblasColMajor, CblasNoTrans, CblasTrans,3,3,2,1,A, 3, B, 3,2,C,3);

  for(i=0; i<9; i++)
    printf("%lf ", float(C[i]));
  printf("\n");

  return 0;
}
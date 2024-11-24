#include <stdio.h>
#include <math.h>
#include <cyth.h>

int main()
{
  char a = 92, b = 56, c = 61, d = 4;
  float combine = a * pow(128, 0) + b * pow(128, 1) + c * pow(128, 2) + d * pow(128, 3);
  printf("%f\n", combine);
  float small = sqrt(combine);
  printf("%f\n", small);
  printf("%f\n", sqrt(small));
  float stored = 55.364014;
  float extracted = pow(stored, 4);
  printf("%f\n", extracted);
  char _d = extracted / pow(128, 3);
  printf("d: %d; ", _d);
  extracted -= _d * pow(128, 3);
  char _c = extracted / pow(128, 2);
  printf("c: %d; ", _c);
  extracted -= _c * pow(128, 2);
  char _b = extracted / pow(128, 1);
  printf("b: %d; ", _b);
  extracted -= _b * pow(128, 1);
  char _a = extracted / pow(128, 0);
  printf("a: %d; ", _a);
  return 0;
}
#pragma once
#include <stdarg.h>
#include <cyth.h>
#include <str.h>

typedef struct tensor
{
  uchar ndim;
  ushort *shape;
  uint size;
  float *data;
} tensor;

void tensor_init(tensor *aten, ...)
{
  va_list args;
  va_start(args, aten);
  aten->size = 1, aten->shape = nalloc(ushort, aten->ndim);
  for_m(uchar, aten->ndim, i) aten->shape[i] =
      va_arg(args, unsigned),
                              aten->size *= aten->shape[i];
  aten->data = nalloc(float, aten->size), va_end(args);
}
uint index_by(const uchar ndim, const ushort *shape, ...)
{
  uint index = 0;
  va_list indices;
  va_start(indices, shape);
  for_m(uchar, ndim, i) index *= shape[i], index += va_arg(indices, unsigned);
  va_end(indices);
  return index;
}

#define data_of(aten, ...) (aten).data[index_by((aten).ndim, (aten).shape, __VA_ARGS__)]
#define mat_at(mat, i, j) (mat).data[(i) * (mat).shape[1] + (j)]
#define at_3d(aten, i, j, k) (aten).data[(aten).shape[1] * ((i) + (aten).shape[2] * (j)) + (k)]
#define with_ndim(ndim) = {ndim}
#define copy(arr, to, n) for_n(n, i)(to)[i] = (arr)[i]

void deten(tensor *aten) { free(aten->data), free(aten->shape); }
void shapen(tensor *aten, const ushort *shape)
{
  aten->shape = nalloc(ushort, aten->ndim);
  copy(shape, aten->shape, aten->ndim);
  aten->size = product(uint, shape[i], for_n(aten->ndim, i));
  aten->data = nalloc(float, aten->size);
}

#include <stdio.h>

void randn(tensor *aten) { for_n(aten->size, i) aten->data[i] = randf(-1, 1); }

void print_nd_arr(
    const float *restrict data, const ushort *restrict shape,
    const uint size, const uchar ndim, const uchar shift)
{
  const uint batch_n = size / *shape;
  printf("{");
  if (ndim == 1)
  {
    printf("%.2f", *data);
    for_fn(1, *shape, i) printf(", %.2f", data[i]);
  }
  else
  {
    print_nd_arr(data, shape + 1, batch_n, ndim - 1, shift + 1), data += batch_n;
    for (const float *restrict l = data + size; data + (batch_n) != l; data += batch_n)
    {
      printf(",\n");
      for_n(shift, i) printf(" ");
      print_nd_arr(data, shape + 1, batch_n, ndim - 1, shift + 1);
    }
  }
  printf("}");
}
void print_tensor(const tensor aten)
{
  const float *data = aten.data;
  print_nd_arr(data, aten.shape, aten.size, aten.ndim, 1);
}

void writen(const tensor aten, FILE *file)
{
  fprintf(file, "%d ", aten.ndim);
  for_n(aten.ndim, i) fprintf(file, "%d ", aten.shape[i]);
  for_n(aten.size, i) fprintf(file, "%f ", aten.data[i]);
}
tensor readten(FILE *file)
{
  tensor result;
  fscanf(file, "%d", &result.ndim);
  result.shape = nalloc(ushort, result.ndim);
  for_n(result.ndim, i) fscanf(file, "%d", &result.shape[i]);
  result.size = product(uint, result.shape[i], for_n(result.ndim, i));
  result.data = nalloc(float, result.size);
  for_n(result.size, i) fscanf(file, "%f", &result.data[i]);
  return result;
}

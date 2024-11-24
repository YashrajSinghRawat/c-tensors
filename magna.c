#include <time.h>
#include "magma.c"

void next_indices();

tensor transpose_n(const tensor aten, ...)
{
  // get the axes from va args
  va_list args;
  va_start(args, aten);
  ushort axes[aten.ndim];
  for_n(aten.ndim, i) axes[i] = (ushort)va_arg(args, int);
  va_end(args);

  // arrage the axes in dimensions
  ushort dims[aten.ndim];
  for_n(aten.ndim, i) dims[i] = aten.shape[axes[i]];

  // prepare the result
  tensor result = scale(aten);
  reshape(&result, dims);

  // use next_index to transpose each batch separately
  return result;
}

int main()
{
  tensor a with_ndim(3);
  tensor_init(&a, 2, 3, 4);
  for_n(a.size, i) a.data[i] = i + 1;
  print_tensor(a);
  return 0;
}
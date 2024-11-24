#pragma once
#include <math.h>
#include "tensor.c"

float sum_arr(const float *arr, const uint n)
{
  float e = 0;
  for (uint i = 0; i < n;)
    e += arr[i++];
  return e;
}
#define mean_arr(arr, n) (sum_arr(arr, n) / (n))
float var_arr(const float *arr, const uint n)
{
  float mean = mean_arr(arr, n), sd[n];
  for_n(n, i) sd[i] = powf(arr[i] - mean, 2);
  return mean_arr(sd, n);
}
float max_arr(const float *arr, const uint n)
{
  float m = -9e99;
  for_fn(1, n, i) m = fmax(arr[i], m);
  return m;
}
float min_arr(const float *arr, const uint n)
{
  float m = 9e99;
  for_fn(1, n, i) m = fmin(arr[i], m);
  return m;
}

tensor transpose(const tensor mat)
{
  tensor result = {2, malloc(4), mat.size, malloc(4 * mat.size)};
  *result.shape = mat.shape[1], result.shape[1] = *mat.shape;
  const uint m = *mat.shape, n = mat.shape[1], p = mat.size;
  static uint i, j, o;
  const uint _size = p - 1;
  for (i = j = o = 0; i < p; j -= _size)
    for (o += n; i < o; j += m, ++i)
      result.data[j] = mat.data[i];
  return result;
}

tensor matmul(const tensor a, const tensor b)
{
  uint i, j, k, l;
  const ushort m = *a.shape, n = *b.shape, o = b.shape[1];
  const uint size = m * o, bn = b.size - 1;
  tensor c = {2, malloc(4), size, malloc(4 * size)};
  *c.shape = m, c.shape[1] = o;
  for (*c.data = j = i = 0; i < size; j += n)
    for (l = 0; l < o; ++i, l -= bn, c.data[i] = 0)
      for (k = j; l < b.size; ++k, l += o)
        c.data[i] += a.data[k] * b.data[l];
  return c;
}
tensor sum_ten(const tensor aten, const uchar axis)
{
  tensor e with_ndim(aten.ndim - axis);
  shapen(&e, aten.shape);
  const uint batch_n = aten.size / e.size;
  for (uint i = 0; i < e.size; i++)
    e.data[i] = sum_arr(&aten.data[i * batch_n], batch_n);
  return e;
}
tensor mean(const tensor aten, const uchar axis)
{
  tensor m with_ndim(aten.ndim - axis);
  shapen(&m, aten.shape);
  const uint batch_n = aten.size / m.size;
  for_n(m.size, i) m.data[i] = mean_arr(&aten.data[i * batch_n], batch_n);
  return m;
}
tensor variance(const tensor aten, const uchar axis)
{
  tensor v with_ndim(aten.ndim - axis);
  shapen(&v, aten.shape);
  const uint batch_n = aten.size / v.size;
  for_n(v.size, i) v.data[i] = var_arr(&aten.data[i * batch_n], batch_n);
  return v;
}
tensor max_ten(const tensor aten, const uchar axis)
{
  tensor m with_ndim(aten.ndim - axis);
  shapen(&m, aten.shape);
  const uint batch_n = aten.size / m.size;
  for_n(m.size, i) m.data[i] = max_arr(&aten.data[i * batch_n], batch_n);
  return m;
}
tensor min_ten(const tensor aten, const uchar axis)
{
  tensor m with_ndim(aten.ndim - axis);
  shapen(&m, aten.shape);
  const uint batch_n = aten.size / m.size;
  for_n(m.size, i) m.data[i] = min_arr(&aten.data[i * batch_n], batch_n);
  return m;
}

void add_ten(tensor *a, const tensor b) { for_n(a->size, i) a->data[i] += b.data[i]; }
void sub_ten(tensor *a, const tensor b) { for_n(a->size, i) a->data[i] -= b.data[i]; }
void mul_ten(tensor *a, const tensor b) { for_n(a->size, i) a->data[i] *= b.data[i]; }
void div_ten(tensor *a, const tensor b) { for_n(a->size, i) a->data[i] /= b.data[i]; }

void add_val(tensor *aten, const float a) { for_n(aten->size, i) aten->data[i] += a; }
void sub_val(tensor *aten, const float a) { for_n(aten->size, i) aten->data[i] -= a; }
void mul_val(tensor *aten, const float a) { for_n(aten->size, i) aten->data[i] *= a; }
void div_val(tensor *aten, const float a) { for_n(aten->size, i) aten->data[i] /= a; }

tensor scale(const tensor aten)
{
  tensor aten2 with_ndim(aten.ndim);
  shapen(&aten2, aten.shape);
  for_n(aten.size, i) aten2.data[i] = aten.data[i];
  return aten2;
}
void reshape(tensor *aten, ...)
{
  va_list args;
  va_start(args, aten);
  for_m(uchar, aten->ndim, i) aten->shape[i] = va_arg(args, unsigned);
  va_end(args);
}

#define T transpose

void posenc(tensor *aten)
{
  for_n(aten->shape[0], i) for_n(aten->shape[1], j) if (j % 2)
      mat_at(*aten, i, j) = cosf(i / powf(1e4, 2.0 * (j - 1) / aten->shape[1]));
  else mat_at(*aten, i, j) = sinf(i / powf(1e4, 2.0 * j / aten->shape[1]));
}
/// @brief To split a 2d tensor into n-pieces.
/// @param aten: is the tensor which is being pieced.
/// @param tens: is the obtained pieces of tensor.
/// @param n: is the number of elements in each piece.
tensor *split_ten(const tensor aten, const uint n)
{
  const ushort split_n = aten.shape[0] % n ? aten.shape[0] / n + 1 : aten.shape[0] / n;
  tensor *tens = nalloc(tensor, split_n);
  uint pre_index = 0;

  for_n(split_n, i)
  {
    const uint split_index = fmin(n * (i + 1), aten.shape[0]);
    const uint diff_index = split_index - pre_index;
    tens[i].ndim = 2, tensor_init(&tens[i], diff_index, aten.shape[1]);

    for_n(diff_index, j) for_n(aten.shape[1], k)
        mat_at(tens[i], j, k) = mat_at(aten, pre_index + j, k);
    pre_index = split_index;
  }
  return tens;
}

float mse_loss(const tensor outputs, const tensor targets)
{
  float loss = 0;
  for_n(outputs.size, i) loss +=
      (outputs.data[i] - targets.data[i]) * (outputs.data[i] - targets.data[i]);
  return loss / outputs.size;
}
tensor gradient(const tensor outputs, const tensor targets)
{
  tensor grad with_ndim(outputs.ndim);
  shapen(&grad, outputs.shape);
  for_n(outputs.size, i) grad.data[i] =
      2 * (outputs.data[i] - targets.data[i]) / outputs.shape[0];
  return grad;
}
float scaled_max_loss(const tensor outputs, const tensor targets)
{
  float loss = -9e99, temp, scaled = 0;
  for_n(outputs.size, i)
  {
    temp = outputs.data[i] - targets.data[i];
    temp = sqrtf(temp * temp);
    loss = fmax(loss, temp);
    scaled += sqrtf(targets.data[i] * targets.data[i]);
  }
  return loss * 1e2 / (scaled / outputs.size);
}
float scaled_loss(const tensor outputs, const tensor targets)
{
  float loss = 0, temp, scaled = 0;
  for_n(outputs.size, i)
  {
    temp = outputs.data[i] - targets.data[i];
    loss += sqrtf(temp * temp);
    scaled += sqrtf(targets.data[i] * targets.data[i]);
  }
  return loss * 1e2 / scaled;
}
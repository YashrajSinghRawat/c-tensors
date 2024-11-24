#pragma once
#include <math.h>
#include "../magma.c"

void sigmoid(
    tensor *aten) { for_n((*aten).size, i)(*aten).data[i] = 1. / (1. + exp2f(-(*aten).data[i])); }
void sigmoid_d(tensor *aten) { for_n((*aten).size, i)(*aten).data[i] *= 1. - (*aten).data[i]; }

void tanh_ten(tensor *aten) { for_n((*aten).size, i)(*aten).data[i] = tanhf((*aten).data[i]); }
void tanh_d(tensor *aten) { for_n((*aten).size, i)(*aten).data[i] = 1 - powf((*aten).data[i], 2.); }

void relu(tensor *aten) { for_n((*aten).size, i)(*aten).data[i] = fmax(0, (*aten).data[i]); }
void relu_d(tensor *aten) { for_n((*aten).size, i)(*aten).data[i] = (*aten).data[i] > 0; }

void leaky_relu(tensor *aten, const float alpha)
{
  for_n((*aten).size, i)(*aten).data[i] = (*aten).data[i] > 0 ? (*aten).data[i] : (*aten).data[i] * alpha;
}
void leaky_relu_d(tensor *aten, const float alpha)
{
  for_n((*aten).size, i)(*aten).data[i] = (*aten).data[i] > 0 ? (*aten).data[i] : alpha;
}
void elu(tensor *aten, const float alpha)
{
  for_n((*aten).size, i)(*aten).data[i] = (*aten).data[i] > 0 ? (*aten).data[i] : alpha * ((*aten).data[i] - 1);
}
void elu_d(tensor *aten, const float alpha)
{
  for_n((*aten).size, i)(*aten).data[i] = (*aten).data[i] > 0 ? 1 : alpha * exp2f((*aten).data[i]);
}

// void softmax(tensor *aten)
// {
//   tensor result = max_ten(T(*aten), 1);
//   for_n(result.size, i)
//   {
//     result.data[i] = exp(result.data[i]);
//     result.data[i] /= result.data[i] * aten->shape[aten->ndim - 1] + 1e-10;
//     if (result.data[i] == NAN)
//       result.data[i] = 1;
//   }

//   for_n(aten->size, i) copy(result.data, &mat_at(*aten, i, 0), result.size);
// }

#define _sigmoid 0
#define _tanh 1
// Removes negative numbers.
#define _relu 2
#define _leaky_relu 3
#define _elu 4
#define _softmax 5

void activate(tensor *inputs, const uchar act)
{
  switch (act)
  {
  case _sigmoid:
    return sigmoid(inputs);
  case _tanh:
    return tanh_ten(inputs);
  case _relu:
    return relu(inputs);
  case _leaky_relu:
    return leaky_relu(inputs, 1e-3);
  case _elu:
    return elu(inputs, 1);
  default:
    break;
  }
}
void activate_d(tensor *grads, const uchar act)
{
  switch (act)
  {
  case _sigmoid:
    return sigmoid_d(grads);
  case _tanh:
    return tanh_d(grads);
  case _relu:
    return relu_d(grads);
  case _leaky_relu:
    return leaky_relu_d(grads, 1e-3);
  case _elu:
    return elu_d(grads, 1);
  default:
    break;
  }
}

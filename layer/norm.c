#pragma once
#include <float.h>
#include "../magma.c"

typedef struct normal_layer
{
  float lnr;
  tensor means, vars, scales, biases, inputs;
} normal_layer;

#define init_nol(nl, lnr_, n)                                    \
  nl.lnr = lnr_, nl.scales.ndim = 1, tensor_init(&nl.scales, n); \
  nl.biases.ndim = 1, tensor_init(&nl.biases, n);

tensor normalize(normal_layer *norm, const tensor inputs)
{
  tensor inputsT = T(inputs);
  norm->inputs = scale(inputs), norm->means = mean(inputsT, 1);
  norm->vars = variance(inputsT, 1), deten(&inputsT);

  tensor outputs with_ndim(2);
  shapen(&outputs, inputs.shape);
  for_n(inputs.shape[0], i) for_n(inputs.shape[1], j)
  {
    mat_at(outputs, i, j) = ((mat_at(inputs, i, j) - norm->means.data[j]) / sqrtf(norm->vars.data[j] + FLT_EPSILON)) * norm->scales.data[j] + norm->biases.data[j];
  }
  return outputs;
}
tensor normalize_d(normal_layer *norm, const tensor outputs_d)
{
  tensor outputs_dT = T(outputs_d), scales_dT;
  tensor scales_d with_ndim(2), biases_d = sum_ten(outputs_dT, 1), inputs_d = scale(outputs_d);
  shapen(&scales_d, outputs_d.shape), deten(&outputs_dT);

  for_n(outputs_d.shape[0], i) for_n(outputs_d.shape[1], j)
  {
    mat_at(scales_d, i, j) = mat_at(outputs_d, i, j) * (mat_at(norm->inputs, i, j) - norm->means.data[j]) / sqrtf(norm->vars.data[j] + FLT_EPSILON);
    mat_at(inputs_d, i, j) /= sqrtf(norm->vars.data[j] + FLT_EPSILON);
  }
  scales_dT = T(scales_d), deten(&scales_d);
  scales_d = sum_ten(scales_dT, 1);
  deten(&scales_dT), deten(&norm->inputs);
  deten(&norm->means), deten(&norm->vars);

  for_n(outputs_d.shape[1], i)
  {
    norm->scales.data[i] -= norm->lnr * scales_d.data[i];
    norm->biases.data[i] -= norm->lnr * biases_d.data[i];
  }
  deten(&scales_d), deten(&biases_d);
  return inputs_d;
}

void nol_write(FILE *file, const normal_layer norm)
{
  writen(norm.scales, file), writen(norm.biases, file);
}
normal_layer nol_cpy(const normal_layer norm)
{
  normal_layer cpy = {norm.lnr};
  cpy.scales = scale(norm.scales), cpy.biases = scale(norm.biases);
  return cpy;
}
void nol_delete(normal_layer *norm) { deten(&norm->scales), deten(&norm->biases); }

// int main()
// {
//   normal_layer norm;
//   init_nol(norm, 1e-1, 3);
//   tensor inputs with_ndim(2);
//   tensor_init(&inputs, 3, 3);
//   mat_at(inputs, 0, 0) = 1, mat_at(inputs, 0, 1) = 2, mat_at(inputs, 0, 2) = 3;
//   mat_at(inputs, 1, 0) = 4, mat_at(inputs, 1, 1) = 5, mat_at(inputs, 1, 2) = 6;
//   mat_at(inputs, 2, 0) = 7, mat_at(inputs, 2, 1) = 8, mat_at(inputs, 2, 2) = 9;

//   tensor outputs = normalize(&norm, inputs);

//   for_n(1e2, i) normalize_d(&norm, gradient(outputs, inputs)),
//       outputs.data = normalize(&norm, inputs).data;

//   print_tensor(outputs);
//   return 0;
// }
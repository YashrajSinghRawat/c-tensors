#pragma once
#include "active.c"

typedef struct dense_layer
{
  tensor weights, biases, inputsT;
  float lnr;
} dense_layer;

#define init_dl(dl, input_n, output_n, lnr_)                                       \
  (dl).weights.ndim = 2, tensor_init(&(dl).weights, input_n, output_n);            \
  randn(&(dl).weights), (dl).biases.ndim = 1, tensor_init(&(dl).biases, output_n); \
  (dl).lnr = lnr_;

tensor dl_forward(dense_layer *dl, const tensor inputs, const uchar activation)
{
  tensor outputs with_ndim(2);
  tensor_init(&outputs, inputs.shape[0], dl->weights.shape[1]);
  dl->inputsT = T(inputs);
  for_n(inputs.shape[0], i) for_n(dl->weights.shape[1], j)
  {
    mat_at(outputs, i, j) = dl->biases.data[j];
    for_n(inputs.shape[1], k) mat_at(outputs, i, j) +=
        mat_at(inputs, i, k) * mat_at(dl->weights, k, j);
  }
  activate(&outputs, activation);
  return outputs;
}
tensor dl_backward(dense_layer *dl, tensor outputs_d, const uchar activation)
{
  activate_d(&outputs_d, activation);
  tensor weight_d = matmul(dl->inputsT, outputs_d);
  tensor outputs_dT, biases_d = sum_ten(outputs_dT = T(outputs_d), 1);

  // updating weights and biases using gradients
  mul_val(&weight_d, dl->lnr), mul_val(&biases_d, dl->lnr);
  sub_ten(&dl->weights, weight_d), sub_ten(&dl->biases, biases_d);
  deten(&weight_d), deten(&biases_d), deten(&outputs_dT);
  tensor weightsT, inputs_d = matmul(outputs_d, weightsT = T(dl->weights));
  deten(&dl->inputsT), deten(&weightsT);
  return inputs_d;
}

void dl_delete(dense_layer *dl) { deten(&dl->weights), deten(&dl->biases); }
dense_layer dl_cpy(const dense_layer dl)
{
  dense_layer cpy = {scale(dl.weights), scale(dl.biases)};
  cpy.lnr = dl.lnr;
  return cpy;
}
void dl_write(FILE *file, const dense_layer dl) { writen(dl.weights, file), writen(dl.biases, file); }

// int main()
// {
//   dense_layer dl;
//   init_dl(dl, 2, 2, 1e-2);

//   tensor inputs with_ndim(2);
//   tensor_init(&inputs, 2, 2);
//   mat_at(inputs, 0, 0) = 1, mat_at(inputs, 0, 1) = 2;
//   mat_at(inputs, 1, 0) = 3, mat_at(inputs, 1, 1) = 4;

//   tensor outputs = dl_forward(&dl, inputs, 5);
//   printf("Before training:"), print_tensor(outputs), printf("\nAfter training:");

//   for_n(1e4, i) sub_ten(&outputs, inputs), dl_backward(&dl, outputs, 5),
//       outputs.data = dl_forward(&dl, inputs, 5).data;
//   print_tensor(outputs);
//   return 0;
// }
#pragma once
#include "../magma.c"

typedef struct linear_layer
{
  tensor weights, biases, inputsT;
  float lnr;
} linear_layer;

#define init_ll(ll, input_n, output_n, lnr_)                                       \
  (ll).weights.ndim = 2, tensor_init(&(ll).weights, input_n, output_n);            \
  randn(&(ll).weights), (ll).biases.ndim = 1, tensor_init(&(ll).biases, output_n); \
  (ll).lnr = lnr_;

tensor ll_forward(linear_layer *ll, const tensor inputs)
{
  tensor outputs with_ndim(2);
  tensor_init(&outputs, inputs.shape[0], ll->weights.shape[1]), ll->inputsT = T(inputs);

  for_n(inputs.shape[0], i) for_n(ll->weights.shape[1], j)
  {
    mat_at(outputs, i, j) = ll->biases.data[j];
    for_n(inputs.shape[1], k) mat_at(outputs, i, j) +=
        mat_at(inputs, i, k) * mat_at(ll->weights, k, j);
  }
  return outputs;
}
tensor ll_backward(linear_layer *ll, const tensor outputD)
{
  tensor weightsT, inputs_d = matmul(outputD, weightsT = T(ll->weights)), gradsT;
  tensor weights_d = matmul(ll->inputsT, outputD), biases_d = sum_ten(gradsT = T(outputD), 1);

  // updating weights and biases using gradients
  mul_val(&weights_d, ll->lnr), mul_val(&biases_d, ll->lnr);
  sub_ten(&ll->weights, weights_d), sub_ten(&ll->biases, biases_d);
  deten(&weightsT), deten(&gradsT), deten(&ll->inputsT);
  deten(&weights_d), deten(&biases_d);
  return inputs_d;
}

void ll_write(FILE *file, const linear_layer ll) { writen(ll.weights, file), writen(ll.biases, file); }
linear_layer ll_cpy(
    const linear_layer ll) { return (linear_layer){scale(ll.weights), scale(ll.biases), ll.lnr}; }
void ll_delete(linear_layer *ll) { deten(&ll->weights), deten(&ll->biases); }

// int main()
// {
//   linear_layer ll;
//   init_ll(ll, 3, 2, 2e-2);

//   tensor inputs with_ndim(2);
//   tensor_init(&inputs, 2, 3);
//   mat_at(inputs, 0, 0) = 1, mat_at(inputs, 0, 1) = 2, mat_at(inputs, 0, 2) = 3;
//   mat_at(inputs, 1, 0) = 4, mat_at(inputs, 1, 1) = 5, mat_at(inputs, 1, 2) = 6;

//   tensor targets with_ndim(2);
//   tensor_init(&targets, 2, 2);
//   mat_at(targets, 0, 0) = 7, mat_at(targets, 0, 1) = 8;
//   mat_at(targets, 1, 0) = 9, mat_at(targets, 1, 1) = 10;
//   const float lnr = 1e-3, in_lnr_step = 1e-5;
//   float min_loss = 9e99, in_lnr = 0;

//   for_n(3e5, i)
//   {
//     linear_layer ll_copy = ll_cpy(ll);
//     ll_copy.lnr = lnr + in_lnr;
//     printf("try with lnr, %f\r", lnr + in_lnr);

//     for_n(1e2, j)
//     {
//       tensor outputs = ll_forward(&ll, inputs);
//       tensor grads = gradient(outputs, targets);
//       ll_backward(&ll, grads), deten(&outputs), deten(&grads);

//       float loss = scaled_loss(outputs, targets);
//       if (loss < min_loss)
//         printf("Epoch is %d, accuracy is %f%% and lnr is %f.\n",
//                j + 1, 100 - (min_loss = loss), lnr + in_lnr);
//     }
//     in_lnr += in_lnr_step;
//     ll_delete(&ll_copy);
//   }

//   return 0;
// }

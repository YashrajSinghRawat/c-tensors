#pragma once
#include "../layer/norm.c"

typedef struct norm_nn
{
  normal_layer *layer;
  uint len;
} norm_nn;

void norm_init(norm_nn *nn, const uint n, const uint len, const float lnr)
{
  nn->len = len, nn->layer = nalloc(normal_layer, len);
  for_n(len, i) { init_nol(nn->layer[i], lnr, n); }
}

tensor norm_forward(norm_nn *nn, const tensor inputs)
{
  tensor outputs = normalize(&nn->layer[0], inputs), outputs_s = scale(outputs);
  deten(&outputs);
  for_fn(1, nn->len, i)
  {
    outputs = normalize(&nn->layer[i], outputs_s), deten(&outputs_s);
    outputs_s = scale(outputs), deten(&outputs);
  }
  return outputs_s;
}
tensor norm_backward(norm_nn *nn, const tensor outputs_d)
{
  tensor inputs_d = normalize_d(&nn->layer[nn->len - 1], outputs_d), inputs_ds = scale(inputs_d);
  deten(&inputs_d);
  rfor_n(nn->len - 1, i)
  {
    inputs_d = normalize_d(&nn->layer[i], inputs_ds);
    deten(&inputs_ds), inputs_ds = scale(inputs_d), deten(&inputs_d);
  }
  return inputs_ds;
}

// int main()
// {
//   norm_nn nn;
//   norm_init(&nn, 200, 2, 1e-1);

//   tensor inputs with_ndim(2);
//   tensor_init(&inputs, 2, 2);
//   mat_at(inputs, 0, 0) = 1, mat_at(inputs, 0, 1) = 2;
//   mat_at(inputs, 1, 0) = 3, mat_at(inputs, 1, 1) = 4;

//   tensor outputs = norm_forward(&nn, inputs);
//   for_n(1e2, _) norm_backward(&nn, gradient(outputs, inputs)),
//       outputs = norm_forward(&nn, inputs);

//   print_tensor(outputs);
//   return 0;
// }
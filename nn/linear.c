#pragma once
#include "../layer/linear.c"

typedef struct linear_nn
{
  linear_layer first, *layer, last;
  ushort model_n;
} linear_nn;

void init_linear(linear_nn *nn, const ushort input_n, const ushort hidden_n,
                 const ushort output_n, const ushort model_d, const float lnr)
{
  init_ll(nn->first, input_n, hidden_n, lnr);
  nn->layer = nalloc(linear_layer, model_d);
  for_n(model_d, i) { init_ll(nn->layer[i], hidden_n, hidden_n, lnr); }
  nn->model_n = model_d, init_ll(nn->last, hidden_n, output_n, lnr);
}

tensor lnn_forward(linear_nn *nn, const tensor inputs)
{
  tensor outputs = ll_forward(&nn->first, inputs), outputs_s = scale(outputs);
  deten(&outputs);
  for_n(nn->model_n, i)
  {
    outputs = ll_forward(&nn->layer[i], outputs_s), deten(&outputs_s);
    outputs_s = scale(outputs), deten(&outputs);
  }
  outputs = ll_forward(&nn->last, outputs_s), deten(&outputs_s);
  return outputs;
}
tensor lnn_backward(linear_nn *nn, const tensor outputD)
{
  tensor layer_d = ll_backward(&nn->last, outputD), layer_s = scale(layer_d);
  deten(&layer_d);
  rfor_n(nn->model_n, i)
  {
    layer_d = ll_backward(&nn->layer[i], layer_s), deten(&layer_s);
    layer_s = scale(layer_d), deten(&layer_d);
  }
  layer_d = ll_backward(&nn->first, layer_s), deten(&layer_s);
  return layer_d;
}

void lnn_write(FILE *file, const linear_nn nn)
{
  fprintf(file, "%lld ", nn.model_n);
  ll_write(file, nn.first), ll_write(file, nn.last);
  for_n(nn.model_n, i) ll_write(file, nn.layer[i]);
}
linear_nn lnn_cpy(const linear_nn lnn)
{
  linear_nn cpy;
  cpy.first = ll_cpy(lnn.first), cpy.last = ll_cpy(lnn.last);
  cpy.layer = nalloc(linear_layer, lnn.model_n), cpy.model_n = lnn.model_n;
  for_n(lnn.model_n, i) cpy.layer[i] = ll_cpy(lnn.layer[i]);
  return cpy;
}
void init_lnn_lnr(linear_nn *lnn, const float lnr)
{
  lnn->first.lnr = lnr, lnn->last.lnr = lnr;
  for_n(lnn->model_n, i) lnn->layer[i].lnr = lnr;
}
void lnn_delete(linear_nn *lnn)
{
  ll_delete(&lnn->first), ll_delete(&lnn->last);
  for_n(lnn->model_n, i) ll_delete(&lnn->layer[i]);
  free(lnn->layer);
}
void lnn_decache(linear_nn *lnn)
{
  deten(&lnn->first.inputsT), deten(&lnn->last.inputsT);
  for_n(lnn->model_n, i) deten(&lnn->layer[i].inputsT);
}

// int main()
// {
//   linear_nn nn;
//   init_linear(&nn, 2, 3, 2, 10, 1e-2);

//   tensor inputs with_ndim(2);
//   tensor_init(&inputs, 2, 2);
//   mat_at(inputs, 0, 0) = 1, mat_at(inputs, 0, 1) = 2;
//   mat_at(inputs, 1, 0) = 3, mat_at(inputs, 1, 1) = 4;

//   tensor outputs = lnn_forward(&nn, inputs);
//   for_n(1e3, i) sub_ten(&outputs, inputs), lnn_backward(&nn, outputs),
//       outputs.data = lnn_forward(&nn, inputs).data;

//   print_tensor(outputs);
//   return 0;
// }
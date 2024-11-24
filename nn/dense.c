#pragma once
#include "../layer/dense.c"

typedef struct dense_nn
{
  dense_layer *layers, first, last;
  uint n, active;
} dense_nn;

void init_dense(
    dense_nn *dnn, const uint inputs_n, const uint hidden_n, const uint outputs_n,
    const uint d_model, const uint active, const float lnr)
{
  dnn->layers = nalloc(dense_layer, d_model), dnn->n = d_model, dnn->active = active;
  init_dl(dnn->first, inputs_n, hidden_n, lnr) init_dl(dnn->last, hidden_n, outputs_n, lnr);
  for_n(d_model, i) { init_dl(dnn->layers[i], hidden_n, hidden_n, lnr) }
}

tensor dnn_forward(dense_nn *dnn, const tensor inputs)
{
  tensor outputs = dl_forward(&dnn->first, inputs, dnn->active), outputs_s = scale(outputs);
  deten(&outputs);
  for_n(dnn->n, i)
  {
    outputs = dl_forward(&dnn->layers[i], outputs_s, dnn->active);
    deten(&outputs_s), outputs_s = scale(outputs), deten(&outputs);
  }
  outputs = dl_forward(&dnn->last, outputs_s, dnn->active), deten(&outputs_s);
  return outputs;
}
tensor dnn_backward(dense_nn *dnn, const tensor outputs_d)
{
  tensor inputs_d = dl_backward(&dnn->last, outputs_d, dnn->active), inputs_ds = scale(inputs_d);
  deten(&inputs_d);
  rfor_n(dnn->n, i)
  {
    inputs_d = dl_backward(&dnn->layers[i], inputs_ds, dnn->active);
    deten(&inputs_ds), inputs_ds = scale(inputs_d), deten(&inputs_d);
  }
  inputs_d = dl_backward(&dnn->first, inputs_ds, dnn->active), deten(&inputs_ds);
  return inputs_d;
}

void dnn_delete(dense_nn *dnn)
{
  dl_delete(&dnn->first), dl_delete(&dnn->last);
  for_n(dnn->n, i) dl_delete(&dnn->layers[i]);
  free(dnn->layers);
}
void dnn_decache(dense_nn *dnn)
{
  deten(&dnn->first.inputsT), deten(&dnn->last.inputsT);
  for_n(dnn->n, i) deten(&dnn->layers[i].inputsT);
}
dense_nn dnn_cpy(const dense_nn dnn)
{
  dense_nn cpy = {
      nalloc(dense_layer, dnn.n), dl_cpy(dnn.first), dl_cpy(dnn.last), dnn.n, dnn.active};
  for_n(dnn.n, i) cpy.layers[i] = dl_cpy(dnn.layers[i]);
  return cpy;
}
void dnn_write(FILE *file, const dense_nn dnn)
{
  fprintf(file, "%d %d ", dnn.n, dnn.active);
  dl_write(file, dnn.first), dl_write(file, dnn.last);
  for_n(dnn.n, i) dl_write(file, dnn.layers[i]);
}
void init_dnn_lnr(dense_nn *dnn, const float lnr)
{
  dnn->first.lnr = lnr, dnn->last.lnr = lnr;
  for_n(dnn->n, i) dnn->layers[i].lnr = lnr;
}

// int main()
// {
//   dense_nn nn;
//   init_dense(&nn, 2, 2, 1, _relu, 1e-3);

//   tensor inputs with_ndim(2);
//   tensor_init(&inputs, 2, 2);
//   mat_at(inputs, 0, 0) = 1, mat_at(inputs, 0, 1) = 2;
//   mat_at(inputs, 1, 0) = 3, mat_at(inputs, 1, 1) = 4;

//   tensor outputs = dnn_forward(&nn, inputs);
//   print_tensor(outputs), printf("\n");

//   for_n(1e2, _) dnn_backward(&nn, gradient(outputs, inputs)),
//       outputs = dnn_forward(&nn, inputs);
//   print_tensor(outputs);
//   return 0;
// }
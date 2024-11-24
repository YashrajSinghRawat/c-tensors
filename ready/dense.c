#pragma once
#include "../magma.c"
#include "active.c"

// FIRST: Make adjustments in the traditional layer and network for speed.

typedef struct dense_layer
{
  tensor weights, biases;
} dense_layer;
#define init_dl(dl, input_n, output_n)                                  \
  (dl).weights.ndim = 2, tensor_init(&(dl).weights, input_n, output_n); \
  randn(&(dl).weights), (dl).biases.ndim = 1, tensor_init(&(dl).biases, output_n);
tensor dl_forward(const dense_layer dl, const tensor inputs, const uint activation)
{
  tensor outputs with_ndim(2);
  tensor_init(&outputs, inputs.shape[0], dl.weights.shape[1]);

  for_n(inputs.shape[0], i) for_n(dl.weights.shape[1], j)
  {
    mat_at(outputs, i, j) = dl.biases.data[j];
    for_n(inputs.shape[1], k) mat_at(outputs, i, j) +=
        mat_at(inputs, i, k) * mat_at(dl.weights, k, j);
  }
  activate(&outputs, activation);
  return outputs;
}

typedef struct dense_nn
{
  dense_layer first, *layer, last;
  uint model_n, active;
} dense_nn;
void init_dense(dense_nn *nn, const ushort input_n, const ushort hidden_n,
                const ushort output_n, const ushort model_d)
{
  init_dl((*nn).first, input_n, hidden_n);
  (*nn).layer = nalloc(dense_layer, model_d);
  for_n(model_d, i) { init_dl((*nn).layer[i], hidden_n, hidden_n); }
  (*nn).model_n = model_d, init_dl((*nn).last, hidden_n, output_n);
}
tensor dnn_forward(const dense_nn nn, const tensor inputs)
{
  tensor outputs = dl_forward(nn.first, inputs, nn.active), outputs_s = scale(outputs);
  deten(&outputs);
  for_n(nn.model_n, i)
  {
    outputs = dl_forward(nn.layer[i], outputs_s, nn.active);
    deten(&outputs_s), outputs_s = scale(outputs), deten(&outputs);
  }
  outputs = dl_forward(nn.last, outputs_s, nn.active), deten(&outputs_s);
  return outputs;
}

// SECOND: Make dl_read and dnn_read function to load the model.

#define dl_read(file) \
  (dense_layer) { readten(file), readten(file) }
dense_nn dnn_read(FILE *file)
{
  dense_nn lnn;
  fscanf(file, "%d%d", &lnn.model_n, &lnn.active);
  lnn.first = dl_read(file), lnn.last = dl_read(file);
  lnn.layer = nalloc(dense_layer, lnn.model_n);
  for_n(lnn.model_n, i) lnn.layer[i] = dl_read(file);
  return lnn;
}

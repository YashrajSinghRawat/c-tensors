#pragma once
#include "../magma.c"

// FIRST: Make adjustments in the traditional layer and network for speed.

typedef struct linear_layer { tensor weights, biases; } linear_layer;
#define init_ll(ll, input_n, output_n)                                       \
  (ll).weights.ndim = 2, tensor_init(&(ll).weights, input_n, output_n);            \
  randn(&(ll).weights), (ll).biases.ndim = 1, tensor_init(&(ll).biases, output_n);
tensor ll_forward(const linear_layer ll, const tensor inputs)
{
  tensor outputs with_ndim(2);
  tensor_init(&outputs, inputs.shape[0], ll.weights.shape[1]);

  for_n(inputs.shape[0], i) for_n(ll.weights.shape[1], j)
  {
    mat_at(outputs, i, j) = ll.biases.data[j];
    for_n(inputs.shape[1], k) mat_at(outputs, i, j) +=
        mat_at(inputs, i, k) * mat_at(ll.weights, k, j);
  }
  return outputs;
}

typedef struct linear_nn { linear_layer first, *layer, last; ushort model_n; } linear_nn;
void init_linear(linear_nn *nn, const ushort input_n, const ushort hidden_n,
  const ushort output_n, const ushort model_d) {
  init_ll((*nn).first, input_n, hidden_n); (*nn).layer = nalloc(linear_layer, model_d);
  for_n(model_d, i) { init_ll((*nn).layer[i], hidden_n, hidden_n); }
  (*nn).model_n = model_d, init_ll((*nn).last, hidden_n, output_n); }
tensor lnn_forward(const linear_nn nn, const tensor inputs)
{ tensor outputs = ll_forward(nn.first, inputs);
  for_n(nn.model_n, i) outputs.data = ll_forward(nn.layer[i], outputs).data;
  return ll_forward(nn.last, outputs); }

// SECOND: Make ll_read and lnn_read function to load the model.

linear_layer ll_read(FILE *file) { linear_layer ll;
  ll.weights = readten(file), ll.biases = readten(file); return ll; }
linear_nn lnn_read(FILE *file) { linear_nn lnn; fscanf(file, "%lld", &lnn.model_n);
  lnn.first = ll_read(file), lnn.last = ll_read(file), lnn.layer = nalloc(linear_layer, lnn.model_n);
  for_n(lnn.model_n, i) lnn.layer[i] = ll_read(file); return lnn; }
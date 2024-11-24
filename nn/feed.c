#pragma once
#include "../layer/feed.c"

typedef struct feed_nn
{
  ff_layer first, *layers, last;
  uint n;
} feed_nn;

void init_feed(
    feed_nn *fnn, const uint inputs_n, const uint outputs_n,
    const uint hidden_n, const float lnr, const uint activation)
{
  const uint trans_n = max(inputs_n, outputs_n);
  init_ff(fnn->first, inputs_n, trans_n, lnr, activation);
  init_ff(fnn->last, trans_n, outputs_n, lnr, activation);
  fnn->layers = nalloc(ff_layer, hidden_n), fnn->n = hidden_n;
  for_n(hidden_n, i) { init_ff(fnn->layers[i], trans_n, trans_n, lnr, activation) }
}

tensor feed_forward(feed_nn *fnn, const tensor inputs)
{
  tensor outputs = ff_forward(&fnn->first, inputs), outputs_s = scale(outputs);
  deten(&outputs);
  for_n(fnn->n, i)
  {
    outputs = ff_forward(&fnn->layers[i], outputs_s), deten(&outputs_s);
    outputs_s = scale(outputs), deten(&outputs);
  }
  outputs = ff_forward(&fnn->last, outputs_s), deten(&outputs_s);
  return outputs;
}
tensor feed_backward(feed_nn *fnn, const tensor grads)
{
  tensor inputs_d = ff_backward(&fnn->last, grads), inputs_ds = scale(inputs_d);
  deten(&inputs_d);
  rfor_n(fnn->n, i)
  {
    inputs_d = ff_backward(&fnn->layers[i], inputs_ds), deten(&inputs_ds);
    inputs_ds = scale(inputs_d), deten(&inputs_d);
  }
  inputs_d = ff_backward(&fnn->first, inputs_ds), deten(&inputs_ds);
  return inputs_d;
}
feed_nn fnn_cpy(const feed_nn fnn)
{
  feed_nn cpy = {ff_cpy(fnn.first), nalloc(ff_layer, fnn.n), ff_cpy(fnn.last), fnn.n};
  for_n(fnn.n, i) cpy.layers[i] = ff_cpy(fnn.layers[i]);
  return cpy;
}
void init_fnn_lnr(feed_nn *fnn, const float lnr)
{
  init_ff_lnr(&fnn->first, lnr), init_ff_lnr(&fnn->last, lnr);
  for_n(fnn->n, i) init_ff_lnr(fnn->layers + i, lnr);
}
void fnn_write(FILE *file, const feed_nn fnn)
{
  fprintf(file, "%d ", fnn.n), ff_write(file, fnn.first), ff_write(file, fnn.last);
  for_n(fnn.n, i) ff_write(file, fnn.layers[i]);
}
void fnn_decache(feed_nn *fnn)
{
  deten(&fnn->first.dense.inputsT), deten(&fnn->first.norm.inputs);
  deten(&fnn->first.norm.means), deten(&fnn->first.norm.vars);
  deten(&fnn->last.dense.inputsT), deten(&fnn->last.norm.inputs);
  deten(&fnn->last.norm.means), deten(&fnn->last.norm.vars);
  for_n(fnn->n, i)
  {
    deten(&fnn->layers[i].dense.inputsT), deten(&fnn->layers[i].norm.inputs);
    deten(&fnn->layers[i].norm.means), deten(&fnn->layers[i].norm.vars);
  }
}
void fnn_delete(feed_nn *fnn)
{
  ff_delete(&fnn->first), ff_delete(&fnn->last);
  for_n(fnn->n, i) ff_delete(&fnn->layers[i]);
}

// int main()
// {
//   tensor inputs with_ndim(2);
//   tensor_init(&inputs, 2, 2);
//   for_n(inputs.size, i) inputs.data[i] = i + 1;

//   feed_nn fnn;
//   init_feed(&fnn, 2, 2, 1, 1e-1, _relu);

//   for_n(1e2, i)
//   {
//     tensor outputs = feed_forward(&fnn, inputs);
//     tensor grads = gradient(outputs, inputs);
//     tensor inputs_d = feed_backward(&fnn, grads);
//     deten(&outputs), deten(&grads), deten(&inputs_d);
//   }

//   print_tensor(feed_forward(&fnn, inputs));
//   return 0;
// }
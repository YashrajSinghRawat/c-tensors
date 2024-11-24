#pragma once
#include "../layer/dense.c"
#include "../layer/norm.c"

typedef struct ff_layer
{
  dense_layer dense;
  normal_layer norm;
  uint activation;
} ff_layer;

#define init_ff(ff, inputs_n, outputs_n, lnr, activation_) \
  init_dl((ff).dense, inputs_n, outputs_n, lnr)            \
      init_nol((ff).norm, lnr, outputs_n)(ff)              \
          .activation = activation_;

tensor ff_forward(ff_layer *ff, const tensor inputs)
{
  tensor d_outputs = dl_forward(&ff->dense, inputs, ff->activation);
  tensor n_outputs = normalize(&ff->norm, d_outputs);
  deten(&d_outputs);
  return n_outputs;
}
tensor ff_backward(ff_layer *ff, const tensor outputs_d)
{
  tensor n_inputs_d = normalize_d(&ff->norm, outputs_d);
  tensor d_inputs_d = dl_backward(&ff->dense, n_inputs_d, ff->activation);
  deten(&n_inputs_d);
  return d_inputs_d;
}

void ff_write(FILE *file, const ff_layer ff)
{
  fprintf(file, "%d ", ff.activation);
  dl_write(file, ff.dense), nol_write(file, ff.norm);
}
void init_ff_lnr(ff_layer *ff, const float lnr) { ff->dense.lnr = lnr, ff->norm.lnr = lnr; }
ff_layer ff_cpy(const ff_layer ff) { return (ff_layer){
    dl_cpy(ff.dense), nol_cpy(ff.norm), ff.activation}; }
void ff_delete(ff_layer *ff)
{
  dl_delete(&ff->dense), nol_delete(&ff->norm);
}

// int main()
// {
//   tensor inputs with_ndim(2);
//   tensor_init(&inputs, 2, 2);
//   inputs.data[0] = 1, inputs.data[1] = 2;
//   inputs.data[2] = 3, inputs.data[3] = 4;

//   ff_layer ff;
//   init_ff(ff, 2, 2, 1e-1, 3);

//   for_n(1e2, i)
//   {
//     tensor outputs = ff_forward(&ff, inputs);
//     tensor grads = gradient(outputs, inputs);
//     tensor inputs_d = ff_backward(&ff, grads);
//     deten(&outputs), deten(&grads), deten(&inputs_d);
//   }
//   print_tensor(ff_forward(&ff, inputs));
//   return 0;
// }
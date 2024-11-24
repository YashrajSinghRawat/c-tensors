#pragma once
#include "../layer/self-attn.c"
#include "../layer/norm.c"

typedef struct attn_nn
{
  self_attn *attns, first_al, last_al;
  normal_layer *norms, first_nl, last_nl;
  uint n;
} attn_nn;

void init_Attn(
    attn_nn *ann, const uint input_n, const uint output_n,
    const uint model_d, const float lnr_al, const float lnr_nl)
{
  const uint trans_n = max(input_n, output_n);
  al_init(ann->first_al, input_n, trans_n, lnr_al) init_nol(ann->first_nl, lnr_nl, trans_n);
  al_init(ann->last_al, trans_n, output_n, lnr_al) init_nol(ann->last_nl, lnr_nl, output_n);
  ann->n = model_d, ann->attns = nalloc(self_attn, model_d);
  ann->norms = nalloc(normal_layer, model_d);
  for_n(model_d, i)
  {
    al_init(ann->attns[i], trans_n, trans_n, lnr_al)
        init_nol(ann->norms[i], lnr_nl, trans_n);
  }
}
tensor attn_forward(attn_nn *ann, const tensor inputs)
{
  tensor outputs = al_forward(&ann->first_al, inputs);
  tensor outputs2 = normalize(&ann->first_nl, outputs);
  deten(&outputs);
  for_n(ann->n, i)
  {
    outputs = al_forward(&ann->attns[i], outputs2), deten(&outputs2);
    outputs2 = normalize(&ann->norms[i], outputs), deten(&outputs);
  }
  outputs = al_forward(&ann->last_al, outputs2), deten(&outputs2);
  outputs2 = normalize(&ann->last_nl, outputs), deten(&outputs);
  return outputs2;
}
tensor attn_backward(attn_nn *ann, const tensor grads)
{
  tensor norm_d = normalize_d(&ann->last_nl, grads);
  tensor attn_d = al_backward(&ann->last_al, norm_d);
  deten(&norm_d);
  rfor_n(ann->n, i)
  {
    norm_d = normalize_d(&ann->norms[i], attn_d), deten(&attn_d);
    attn_d = al_backward(&ann->attns[i], norm_d), deten(&norm_d);
  }
  norm_d = normalize_d(&ann->first_nl, attn_d), deten(&attn_d);
  attn_d = al_backward(&ann->first_al, norm_d), deten(&norm_d);
  return attn_d;
}

// int main()
// {
//   tensor inputs with_ndim(2);
//   tensor_init(&inputs, 2, 2);
//   mat_at(inputs, 0, 0) = 1, mat_at(inputs, 0, 1) = 2;
//   mat_at(inputs, 1, 0) = 3, mat_at(inputs, 1, 1) = 4;
//   posenc(&inputs);

//   attn_nn ann;
//   init_Attn(&ann, 2, 2, 1, 1e-11, 8e-3);

//   tensor outputs = attn_forward(&ann, inputs);
//   tensor grads = gradient(outputs, inputs);
//   tensor inputs_d = attn_backward(&ann, grads);

//   float min_loss = 9e99;
//   for_n(1e2, _)
//   {
//     outputs = attn_forward(&ann, inputs);
//     grads = gradient(outputs, inputs);
//     inputs_d = attn_backward(&ann, grads);
//     const float loss = mse_loss(outputs, inputs);
//     if (loss < min_loss)
//     {
//       min_loss = loss;
//       printf("Epoch %d and loss %f\n", _ + 1, loss);
//     }
//   }

//   printf("inputs :");
//   print_tensor(inputs), printf("\noutputs :");
//   print_tensor(outputs), printf("\ninput_d :");
//   print_tensor(inputs_d), printf("\n");
//   return 0;
// }
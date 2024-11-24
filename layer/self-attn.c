#pragma once
#include "../magma.c"

tensor softmax(const tensor aten)
{
  tensor result with_ndim(aten.ndim);
  shapen(&result, aten.shape);

  const uint(64) batch_n = aten.size / aten.shape[0];

  for_n(aten.shape[0], i)
  {
    const uint(64) batch_i = batch_n * i;
    float max_val = aten.data[batch_i];
    for_n(batch_n, j) max_val = max(max_val, aten.data[batch_i + j]);

    float sum_exp = 0;
    for_n(batch_n, j)
    {
      const uint(64) index = batch_i + j;
      const float exp_val = expf(aten.data[index] - max_val);
      result.data[index] = exp_val;
      sum_exp += exp_val;
    }

    for_n(batch_n, j) result.data[batch_i + j] /= sum_exp;
  }

  return result;
}

typedef struct self_attn
{
  tensor W_q, W_k, W_v, B_q, B_k, B_v;
  tensor inputs, attn, query, key, value;
  float learning_rate;
} self_attn;

#define al_init(al, input_n, output_n, lnr)                                 \
  al.W_q.ndim = 2, tensor_init(&al.W_q, output_n, input_n), randn(&al.W_q); \
  al.W_k.ndim = 2, tensor_init(&al.W_k, output_n, input_n), randn(&al.W_k); \
  al.W_v.ndim = 2, tensor_init(&al.W_v, output_n, input_n), randn(&al.W_v); \
  al.B_q.ndim = 1, tensor_init(&al.B_q, output_n);                          \
  al.B_k.ndim = 1, tensor_init(&al.B_k, output_n);                          \
  al.B_v.ndim = 1, tensor_init(&al.B_v, output_n), al.learning_rate = lnr;

tensor al_forward(self_attn *al, const tensor inputs)
{
  al->query = matmul(al->W_q, T(inputs));
  al->key = matmul(al->W_k, T(inputs));
  al->value = matmul(al->W_v, T(inputs));

  for_n(al->query.shape[0], i) for_n(al->query.shape[1], j)
  {
    mat_at(al->query, i, j) += al->B_q.data[i];
    mat_at(al->key, i, j) += al->B_k.data[i];
    mat_at(al->value, i, j) += al->B_v.data[i];
  }

  al->attn = matmul(T(al->query), al->key);
  div_val(&al->attn, sqrtf(al->W_q.shape[0]));
  al->attn = softmax(al->attn), al->inputs = inputs;
  return matmul(al->attn, T(al->value));
}
tensor al_backward(self_attn *al, const tensor output_d)
{
  const tensor value_d = matmul(T(output_d), al->attn);
  tensor attn_d = matmul(T(al->value), T(output_d));

  for_n(al->attn.size, i) attn_d.data[i] *=
      al->attn.data[i] * (1 - al->attn.data[i]);

  const tensor query_d = matmul(al->key, T(attn_d));
  const tensor key_d = matmul(al->query, T(attn_d));

  const tensor W_q_d = matmul(query_d, al->inputs);
  const tensor B_q_d = sum(query_d, 1);
  const tensor W_k_d = matmul(key_d, al->inputs);
  const tensor B_k_d = sum(key_d, 1);
  const tensor W_v_d = matmul(value_d, al->inputs);
  const tensor B_v_d = sum(value_d, 1);

  for_n(al->W_q.shape[0], i)
  {
    for_n(al->W_q.shape[1], j)
    {
      mat_at(al->W_q, i, j) -= mat_at(W_q_d, i, j) * al->learning_rate;
      mat_at(al->W_k, i, j) -= mat_at(W_k_d, i, j) * al->learning_rate;
      mat_at(al->W_v, i, j) -= mat_at(W_v_d, i, j) * al->learning_rate;
    }
    al->B_q.data[i] -= B_q_d.data[i] * al->learning_rate;
    al->B_k.data[i] -= B_k_d.data[i] * al->learning_rate;
    al->B_v.data[i] -= B_v_d.data[i] * al->learning_rate;
  }
  deten(&attn_d);
  tensor inputs_d = matmul(T(value_d), al->W_v);

  return inputs_d;
}

// int main()
// {
//   tensor inputs with_ndim(2);
//   tensor_init(&inputs, 2, 2);
//   inputs.data[0] = 1, inputs.data[1] = 2;
//   inputs.data[2] = 3, inputs.data[3] = 4;
//   posenc(&inputs);

//   self_attn layer;
//   al_init(layer, 2, 2, 1e-2);
//   tensor outputs = al_forward(&layer, inputs);
//   for_n(10, i)
//   {
//     tensor grad = gradient(outputs, inputs);
//     al_backward(&layer, grad);
//     outputs = al_forward(&layer, inputs);
//   }

//   print_tensor(outputs);
//   return 0;
// }
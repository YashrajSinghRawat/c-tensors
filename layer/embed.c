#pragma once
#include "../magma.c"

tensor init_embed(const uint vocab_n, const uint d_model)
{
  tensor embed with_ndim(2);
  tensor_init(&embed, vocab_n, d_model), randn(&embed);
  return embed;
}

tensor embed_with(const tensor embed, const uchar *arr, const uchar n)
{
  tensor outputs with_ndim(2);
  tensor_init(&outputs, n, embed.shape[1]);

  for_n(n, j) copy(&mat_at(embed, arr[j], 0), &mat_at(outputs, j, 0), embed.shape[1]);
  return outputs;
}

// int main()
// {
//   init_embed(embed, 10, 10);
//   uchar arr[] = {1, 2, 3, 4};

//   tensor outputs = embed_with(embed, arr, 4);
//   print_tensor(outputs);
//   return 0;
// }
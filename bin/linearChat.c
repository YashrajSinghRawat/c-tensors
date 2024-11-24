#include "../nn/linear.c"

const char chars[] = " abcdefghijklmnopqrstuvwxyz!?'";

float num_for(const char c) { for_n(30, i) if (chars[i] == c) return i; }
tensor score_for(const char *strs[], const uint(32) n)
{
  tensor scores with_ndim(2);
  tensor_init(&scores, n, 10);
  for_n(n, i)
  {
    const uint(32) str_len = strlen(strs[i]);
    for (uint(32) j = 0; j < 20; j += 2)
    {
      if (j > str_len)
        mat_at(scores, i, j / 2) = 0;
      else if (strs[i][j] == '\0')
        mat_at(scores, i, j / 2) = 0;
      else if (strs[i][j + 1] == '\0')
        mat_at(scores, i, j / 2) = num_for(strs[i][j]) * 30;
      else
        mat_at(scores, i, j / 2) = num_for(strs[i][j]) * 30 + num_for(strs[i][j + 1]);
    }
  }
  // num_for(strs[i][j]) * 27000 + num_for(strs[i][j + 1]) * 900 +
  // num_for(strs[i][j + 2]) * 30 + num_for(strs[i][j + 3]);
  return scores;
}
char *char_for(const float num)
{
  char *char_2 = nalloc(char, 2);
  char_2[0] = remainderf(num, 900);
  char_2[1] = num / 900;
  return char_2;
}
char **strs_for(const tensor aten)
{
  char **strs = nalloc(char *, 10);
  for_n(10, i)
  {
    strs[i] = nalloc(char, 20);
    for (uint(32) j = 0; j < 20; j += 2)
    {
    }
  }
  return strs;
}

int main()
{
  const char *input_strs[] = {
      "hi! how are you?", "are you stupid?", "what can you do?", "who are you?", "what's up!",
      "i'm fine", "i'm yash", "you are not my pc", "my pc can't chat", ""};
  const char *output_strs[] = {
      "i'm good and you?", "no i'm not", "i can chat with you", "i'm your pc", "i'm always up",
      "that's nice", "so what?", "yes i am", "yes it can chat", "hello world!"};

  const tensor inputs = score_for(input_strs, 10),
               targets = score_for(output_strs, 10);

  linear_nn nn;
  init_linear(&nn, 10, 10, 10, 0, 4e-8);

  tensor outputs = lnn_forward(&nn, inputs);

  for_n(2992, i)
  {
    sub_ten(&outputs, targets), lnn_backward(&nn, outputs),
        outputs.data = lnn_forward(&nn, inputs).data;

    printf("Epoch : %d ; The mse error is %f\n", i + 1, mse_loss(lnn_forward(&nn, inputs), targets));
  }

  // print_tensor(inputs), printf("\n");
  // print_tensor(targets), printf("\n");
  print_tensor(outputs), printf("\n");
  char **output_strings = strs_for(outputs);
  for_n(outputs.shape[0], i) printf(output_strings[i]);
  return 0;
}
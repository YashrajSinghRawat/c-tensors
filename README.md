# Neural Network Library in C: Lightweight Neural Architectures

This repository provides a lightweight C-based implementation of various neural network architectures. These implementations are optimized for simplicity and performance, catering to those who want to learn the inner workings of neural networks or embed them in resource-constrained environments.

## Overview

This library includes multiple neural network components implemented in modular and reusable C code. Each module supports key operations such as initialization, forward propagation, backward propagation (for training), and cleanup. Architectures range from simple dense layers to self-attention mechanisms, with modular normalization layers and activation functions.

## Features

1. **Dense Neural Networks (`dense_nn`)**:
   - Fully connected layers with customizable number of inputs, hidden layers, and outputs.
   - Includes forward and backward propagation methods.
   - Support for activation functions and gradient scaling.

2. **Linear Neural Networks (`linear_nn`)**:
   - Similar to `dense_nn` but optimized for purely linear operations.
   - Can model deep linear transformations with forward and backward propagation.

3. **Feedforward Neural Networks (`feed_nn`)**:
   - Includes dense layers with integrated normalization layers.
   - Configurable number of hidden layers, activation functions, and learning rates.

4. **Normalization Networks (`norm_nn`)**:
   - Standalone layer normalization functionality.
   - Supports gradient computation and scaling for deep learning applications.

5. **Self-Attention Networks (`attn_nn`)**:
   - Implements self-attention mechanisms for sequence modeling.
   - Includes layer normalization and support for multi-layer attention.

## File Structure

- `nn/dense.c`: Implementation of dense neural networks.
- `nn/linear.c`: Implementation of linear neural networks.
- `nn/feed.c`: Feedforward neural network module combining dense and normalization layers.
- `nn/norm.c`: Standalone normalization layers.
- `nn/self-attn.c`: Self-attention mechanism with normalization support.

## Code Example

Below is a sample usage of the `dense_nn` module:

```c
#include "nn/dense.c"

int main() {
    dense_nn nn;
    init_dense(&nn, 2, 2, 1, 2, _relu, 1e-3);

    tensor inputs with_ndim(2);
    tensor_init(&inputs, 2, 2);
    mat_at(inputs, 0, 0) = 1, mat_at(inputs, 0, 1) = 2;
    mat_at(inputs, 1, 0) = 3, mat_at(inputs, 1, 1) = 4;

    tensor outputs = dnn_forward(&nn, inputs);
    print_tensor(outputs);

    for (int i = 0; i < 100; i++) {
        dnn_backward(&nn, gradient(outputs, inputs));
        outputs = dnn_forward(&nn, inputs);
    }

    print_tensor(outputs);
    dnn_delete(&nn);
    return 0;
}
```

### Output:
```plaintext
Initial output tensor:
...
Final output tensor after training:
...
```

## Installation

To use this library:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/c-neural-networks.git
   ```
2. Include the relevant `.c` files in your project.
3. Use a C compiler (e.g., `gcc`) to compile your program.

## Documentation

### Dense Neural Networks (`dense_nn`)

- **Initialization**: 
  ```c
  init_dense(dense_nn *nn, uint inputs_n, uint hidden_n, uint outputs_n, uint d_model, uint active, float lnr);
  ```
  Initializes a dense neural network.

- **Forward Propagation**:
  ```c
  tensor dnn_forward(dense_nn *nn, tensor inputs);
  ```
  Performs forward propagation.

- **Backward Propagation**:
  ```c
  tensor dnn_backward(dense_nn *nn, tensor outputs_d);
  ```
  Computes gradients and updates weights.

### Other Modules

Each module (`linear_nn`, `feed_nn`, `norm_nn`, `attn_nn`) follows a similar structure with functions for initialization, forward/backward propagation, and cleanup.

## Performance

This library is designed for performance on small-scale neural networks. It uses efficient memory allocation and deallocation routines. For heavy computational tasks, consider integrating with libraries like OpenBLAS or Intel MKL.

## Contributions

Contributions are welcome! If you'd like to add new features or optimize existing ones, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

**Author:** Yashraj Singh Rawat
**Contact:** Helloicognitoworld@gmail.com

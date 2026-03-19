# Tensor Operations

> Tensors are the common language between data and deep learning. Every image, every sentence, every gradient flows through them.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~120 minutes

## Why Tensors

Every deep learning framework speaks tensors. Not matrices. Not arrays. Tensors. The reason is that real-world data does not fit into two dimensions.

A single grayscale image is a 2D grid of pixel intensities. Add color channels and it becomes 3D. Batch 32 of those images together for training and it becomes 4D. A language model processes a batch of sequences, each token represented by an embedding vector. That is 3D. Self-attention adds a head dimension, making it 4D.

Matrices handle pairwise relationships between two sets of things. Tensors handle relationships between any number of sets of things. They are the minimal data structure that can represent all the shapes deep learning needs.

## What a Tensor Is

A tensor is a multi-dimensional array of numbers with a uniform data type. The number of dimensions is called the **rank** or **order**.

```
Rank 0: Scalar          3.14                          shape: ()
Rank 1: Vector          [1, 2, 3]                     shape: (3,)
Rank 2: Matrix          [[1, 2], [3, 4]]              shape: (2, 2)
Rank 3: 3D Tensor       [[[1,2],[3,4]], [[5,6],[7,8]]]  shape: (2, 2, 2)
Rank 4: 4D Tensor       A batch of color images        shape: (B, C, H, W)
```

Each dimension is called an **axis**. The shape is a tuple listing the size along each axis. The total number of elements is the product of all sizes in the shape.

```
shape = (2, 3, 4)
total elements = 2 * 3 * 4 = 24
```

Mathematically, a rank-n tensor over a vector space V is a multilinear map from n copies of V (or its dual) to the scalars. In practice, for deep learning, you can think of it as an n-dimensional grid of numbers with defined operations.

## Shapes and Axes in Deep Learning

Different data types map to specific tensor shapes by convention. Knowing these conventions is essential for debugging shape errors, which are the most common bugs in deep learning code.

### Vision: (B, C, H, W)

```
B = batch size        how many images processed together
C = channels          3 for RGB, 1 for grayscale, 64 for a conv layer output
H = height            spatial rows
W = width             spatial columns
```

A batch of 32 RGB images of size 224x224:

```
shape: (32, 3, 224, 224)
total elements: 32 * 3 * 224 * 224 = 4,816,896
```

PyTorch uses NCHW (channels-first). TensorFlow defaults to NHWC (channels-last). This matters for performance. Hardware accelerators have preferences, and mismatched layouts cause silent slowdowns or errors.

### NLP: (B, T, D)

```
B = batch size        how many sequences
T = sequence length   number of tokens (words, subwords, characters)
D = embedding dim     vector size representing each token
```

A batch of 16 sentences, each 128 tokens, each token embedded as a 768-dimensional vector:

```
shape: (16, 128, 768)
total elements: 16 * 128 * 768 = 1,572,864
```

### Attention: (B, H, T, D)

```
B = batch size
H = number of attention heads
T = sequence length
D = head dimension (usually embedding_dim / num_heads)
```

BERT with 12 heads, 768-dim embeddings: each head gets 768/12 = 64 dimensions.

```
Q, K, V shape: (B, 12, T, 64)
```

### Audio: (B, C, F, T)

```
B = batch size
C = channels (1 for mono, 2 for stereo)
F = frequency bins (e.g., 128 mel bands)
T = time frames
```

### Weights

Neural network weight tensors also follow conventions:

```
Linear layer:        (out_features, in_features)
Conv2D kernel:       (out_channels, in_channels, kernel_h, kernel_w)
Embedding table:     (vocab_size, embedding_dim)
LayerNorm:           (normalized_shape,)
```

## Reshaping Operations

Reshaping changes how elements are arranged into dimensions without changing the elements themselves. The total number of elements must stay the same.

### Reshape

```
Original: shape (2, 6) with 12 elements
 [[0, 1, 2, 3, 4, 5],
  [6, 7, 8, 9, 10, 11]]

Reshape to (3, 4):
 [[0, 1, 2, 3],
  [4, 5, 6, 7],
  [8, 9, 10, 11]]

Reshape to (12,):
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

Reshape to (2, 2, 3):
 [[[0, 1, 2],
   [3, 4, 5]],
  [[6, 7, 8],
   [9, 10, 11]]]
```

One dimension can be -1, meaning "infer this size from the others."

```
shape (2, 6) reshaped to (-1, 3) gives (4, 3)
because 12 / 3 = 4
```

### Squeeze and Unsqueeze

Squeeze removes dimensions of size 1. Unsqueeze adds a dimension of size 1 at a specified position.

```
shape (1, 3, 1, 5)  after squeeze()      -> (3, 5)
shape (1, 3, 1, 5)  after squeeze(dim=0) -> (3, 1, 5)
shape (3, 5)         after unsqueeze(0)   -> (1, 3, 5)
shape (3, 5)         after unsqueeze(1)   -> (3, 1, 5)
shape (3, 5)         after unsqueeze(-1)  -> (3, 5, 1)
```

Unsqueezing is critical for broadcasting. If you have a bias vector of shape (D,) and want to add it to a batch of shape (B, T, D), you unsqueeze to (1, 1, D) so broadcasting aligns the last axis.

### Transpose and Permute

Transpose swaps two axes. Permute reorders all axes.

```
shape (B, T, D) transposed at (1, 2) gives (B, D, T)
shape (B, C, H, W) permuted as (0, 2, 3, 1) gives (B, H, W, C)
```

This is how you convert between PyTorch's NCHW and TensorFlow's NHWC:

```
NCHW: (B, C, H, W)
permute(0, 2, 3, 1) -> NHWC: (B, H, W, C)
permute(0, 3, 1, 2) -> NCHW: (B, C, H, W)
```

### Flatten

Flatten collapses multiple dimensions into one. In a CNN, after convolution layers produce a (B, C, H, W) tensor, you flatten the last three dimensions to get (B, C*H*W) before feeding into a fully connected layer.

```
shape (32, 64, 7, 7) flattened from dim 1 -> (32, 3136)
```

### View vs Reshape

In PyTorch, `view` requires the tensor to be contiguous in memory. `reshape` works on any tensor (it may copy data if needed). After a transpose or permute, the tensor is typically non-contiguous, so `view` fails but `reshape` succeeds. You can call `.contiguous()` to make a copy that is contiguous, then use `view`.

## Broadcasting

Broadcasting lets you perform operations between tensors of different shapes without explicitly copying data. NumPy, PyTorch, and TensorFlow all follow the same rules.

### The Rules

Two dimensions are compatible when:

1. They are equal, or
2. One of them is 1

Broadcasting aligns shapes from the right (trailing dimensions) and works outward. If one tensor has fewer dimensions, it is padded with 1s on the left.

```
Tensor A:     (8, 1, 6, 1)
Tensor B:        (7, 1, 5)

Step 1 - Pad B:  (1, 7, 1, 5)

Step 2 - Check each dimension:
  dim 0:  8 vs 1  -> broadcast to 8
  dim 1:  1 vs 7  -> broadcast to 7
  dim 2:  6 vs 1  -> broadcast to 6
  dim 3:  1 vs 5  -> broadcast to 5

Result: (8, 7, 6, 5)
```

### Common Broadcasting Patterns

**Adding bias to every sample in a batch:**

```
activations: (B, D)       = (32, 768)
bias:        (D,)         = (768,)
Padded bias: (1, 768)
Result:      (32, 768)    each row gets the same bias added
```

**Scaling each channel of an image batch:**

```
images: (B, C, H, W)     = (32, 3, 224, 224)
scale:  (C, 1, 1)         = (3, 1, 1)
Padded: (1, 3, 1, 1)
Result: (32, 3, 224, 224) each channel scaled by its factor
```

**Outer product via broadcasting:**

```
a: (M, 1)
b: (1, N)
a * b: (M, N)   every element a[i] multiplied with every element b[j]
```

### When Broadcasting Fails

```
Tensor A: (3, 4)
Tensor B: (5,)

dim -1: 4 vs 5  -> not equal, neither is 1 -> ERROR
```

Broadcasting errors produce messages like "shapes (3, 4) and (5,) are not broadcastable." When you see these, check your shapes from right to left.

## Memory Layout

### Row-Major vs Column-Major

A 2D array stored in memory is a 1D sequence of bytes. The question is which dimension is contiguous.

```
Matrix: [[a, b, c],
         [d, e, f]]

Row-major (C order):    [a, b, c, d, e, f]
  Rows are contiguous. Move along last axis = contiguous access.

Column-major (F order): [a, d, b, e, c, f]
  Columns are contiguous. Move along first axis = contiguous access.
```

NumPy defaults to row-major (C order). MATLAB and Fortran use column-major (F order). This matters for performance: iterating along contiguous dimensions is fast because it uses CPU cache lines efficiently.

### Strides

Strides tell you how many bytes (or elements) to skip to move one step along each dimension.

```
shape (3, 4), row-major, dtype=float32 (4 bytes each)

strides = (16, 4)
  To move one step in dim 0 (next row): skip 16 bytes (4 elements * 4 bytes)
  To move one step in dim 1 (next col): skip 4 bytes (1 element * 4 bytes)
```

Transpose does not move data. It swaps the strides:

```
Original:    shape (3, 4), strides (4, 1)   (element strides)
Transposed:  shape (4, 3), strides (1, 4)   (non-contiguous)
```

This is why transposed tensors are non-contiguous: the elements for a row are no longer adjacent in memory.

### Contiguous vs Non-Contiguous

A contiguous tensor has elements stored in a single block of memory with no gaps or reordering. Operations that change the logical order without moving data (transpose, permute, narrow, expand) create non-contiguous views.

Non-contiguous tensors:
- May be slower for element-wise operations
- Cannot use `view` in PyTorch (use `reshape` or call `.contiguous()` first)
- May cause issues with certain CUDA kernels

Check contiguity: `tensor.is_contiguous()` in PyTorch, `array.flags['C_CONTIGUOUS']` in NumPy.

## Einsum: The Universal Tensor Operation

Einstein summation notation (einsum) can express almost any tensor operation in a single line. The idea: label each axis with a letter. Axes that appear in the input but not the output get summed over. Axes that appear in both are kept.

### Syntax

```
result = einsum("input_subscripts -> output_subscripts", tensor_a, tensor_b, ...)
```

Each tensor gets a set of subscript letters. Letters that appear in the output are kept; letters that disappear are summed.

### Building Blocks

**Matrix multiply: ik,kj->ij**

```
A is (I, K), B is (K, J)
k appears in inputs but not output -> sum over k
Result is (I, J)

This is:  C[i,j] = sum_k A[i,k] * B[k,j]
```

**Dot product: i,i->**

```
a is (I,), b is (I,)
i disappears -> sum over i
Result is scalar

This is:  sum_i a[i] * b[i]
```

**Outer product: i,j->ij**

```
a is (I,), b is (J,)
Both i and j appear in output -> keep both
Result is (I, J)

This is:  C[i,j] = a[i] * b[j]
```

**Trace: ii->**

```
A is (N, N)
Repeated index in same tensor = diagonal
i disappears -> sum diagonal

This is:  sum_i A[i,i]
```

**Transpose: ij->ji**

```
A is (I, J)
Both indices kept, order swapped
Result is (J, I)
```

**Batch matrix multiply: bij,bjk->bik**

```
A is (B, I, J), B_tensor is (B, J, K)
b is a batch index (appears in both inputs and output)
j is summed over
Result is (B, I, K)

Each matrix in the batch is multiplied independently.
```

### Einsum for Attention

The core of transformer attention involves several tensor operations that einsum handles cleanly.

**Computing attention scores:**

```
Q: (B, H, T, D)
K: (B, H, T, D)

scores = einsum("bhtd,bhsd->bhts", Q, K)
Result: (B, H, T, T)

For each batch and head, compute dot product between every query
position t and every key position s. The d index is summed.
```

**Applying attention weights to values:**

```
weights: (B, H, T, T)   (after softmax of scores)
V:       (B, H, T, D)

output = einsum("bhts,bhsd->bhtd", weights, V)
Result: (B, H, T, D)

For each batch and head, each output position t is a weighted
sum of value vectors, with weights from the attention matrix.
```

**Multi-head output projection:**

```
heads_output: (B, T, H, D)
W_o:          (H, D, E)       where E = H * D = model dim

output = einsum("bthd,hde->bte", heads_output, W_o)
Result: (B, T, E)
```

### Contraction

Contraction is the general term for summing over shared indices. Matrix multiplication is a contraction. The dot product is a contraction. Trace is a contraction. Any einsum operation where an index disappears from the output is performing a contraction along that axis.

The computational cost of a contraction is proportional to the product of all index sizes (both kept and summed). For `bij,bjk->bik` with B=32, I=128, J=64, K=128, the cost is 32 * 128 * 64 * 128 = 33,554,432 multiply-add operations.

### Outer Product

When no indices are shared between two input tensors, einsum produces an outer product.

```
a: (I,), b: (J,)
einsum("i,j->ij", a, b) produces shape (I, J)

a: (I, J), b: (K, L)
einsum("ij,kl->ijkl", a, b) produces shape (I, J, K, L)
```

The outer product is the building block of rank-1 updates, attention patterns, and factored representations.

## Tensor Operations as Neural Network Operations

Every layer in a neural network is a tensor operation.

```
Operation                  Tensor Form                 Einsum
---------                  -----------                 ------
Linear layer               Y = X @ W.T + b            "bd,od->bo" + bias
Conv2D (as matmul)         unfold + matmul + fold      complex, typically GEMM
Batch norm                 (X - mu) / sigma * gamma    element-wise + broadcast
Attention QKV projection   Q = X @ W_q                "btd,dh->bth"
Attention scores           Q @ K.T / sqrt(d)           "bhtd,bhsd->bhts"
Attention output           softmax(scores) @ V         "bhts,bhsd->bhtd"
Embedding lookup           E[token_ids]                indexing, no einsum
Softmax                    exp(x) / sum(exp(x))        element-wise + reduction
```

### Reductions

Reductions collapse one or more axes by applying an aggregation function.

```
sum along axis:     (B, T, D).sum(dim=1)    -> (B, D)
mean along axis:    (B, T, D).mean(dim=1)   -> (B, D)
max along axis:     (B, T, D).max(dim=1)    -> (B, D) + argmax indices
```

Global average pooling in a CNN takes (B, C, H, W) and averages over H and W to get (B, C). This is `.mean(dim=[2, 3])`.

### Element-wise Operations

Operations that apply independently to each element: ReLU, sigmoid, tanh, GELU, addition, multiplication. These preserve the shape.

```
ReLU:     max(0, x)           shape unchanged
Sigmoid:  1 / (1 + exp(-x))  shape unchanged
Add:      x + y               shapes must broadcast
Multiply: x * y               shapes must broadcast
```

## Practical Shape Debugging

Shape errors account for a large fraction of deep learning bugs. A systematic approach:

1. Print shapes at every step when building a new model
2. Verify that batch dimension is always first and survives through the forward pass
3. After reshape/permute, verify total element count is unchanged
4. When broadcasting, align shapes from the right and check each dimension pair
5. After transpose, remember the tensor is likely non-contiguous

```
Common shape errors and fixes:

Error: mat1 and mat2 shapes cannot be multiplied (32x768 and 512x768)
Fix:   Transpose the weight matrix, or check that input dim matches W's input dim

Error: Expected 4D input (got 3D input)
Fix:   unsqueeze a batch or channel dimension

Error: shape mismatch in concatenation
Fix:   Check that all tensors match in every dimension except the cat dimension
```

## Key Terms

| Term | Definition |
|---|---|
| Tensor | Multi-dimensional array of numbers with uniform type |
| Rank (Order) | Number of dimensions in a tensor |
| Axis (Dimension) | One of the indexing directions of a tensor |
| Shape | Tuple of sizes along each axis |
| Stride | Number of elements to skip to advance one position along an axis |
| Contiguous | Elements stored in sequential memory with no gaps |
| Broadcasting | Automatic expansion of dimensions of size 1 to match a partner tensor |
| Reshape | Change the shape without changing the data order in memory |
| Squeeze | Remove axes of size 1 |
| Unsqueeze | Insert an axis of size 1 at a given position |
| Transpose | Swap two axes |
| Permute | Reorder all axes according to a given sequence |
| Flatten | Collapse multiple axes into a single axis |
| Einsum | Einstein summation notation for expressing tensor operations via index labels |
| Contraction | Summation over a shared index between two tensors |
| Outer product | Tensor formed by multiplying every element of one tensor with every element of another |
| Row-major | Memory layout where the last index changes fastest (C order) |
| Column-major | Memory layout where the first index changes fastest (Fortran order) |
| View | A tensor that shares memory with another, differing only in shape or stride metadata |
| NCHW | Tensor layout: batch, channels, height, width (PyTorch default) |
| NHWC | Tensor layout: batch, height, width, channels (TensorFlow default) |
| Reduction | Operation that collapses one or more axes (sum, mean, max) |

## What Comes Next

With tensors mastered, you can read the shape annotations on any neural network architecture diagram and understand what each operation does to the data. The code file builds a basic Tensor class from scratch, demonstrates broadcasting, walks through einsum examples, and shows common AI tensor shape patterns. Run it, modify the shapes, and build intuition for how data flows through networks.

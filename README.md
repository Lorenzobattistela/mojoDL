# Mojo DL - Deep Learning

This repo aims to build tools to apply deep learning and machine learning techniques using mojo. 
At first, we aim to build the foundatinonal blocks as tensors, layers and simple networks, and then move on to pipelines and other algorithms.

Have in mind that this is a work in progress, and we are still in the process of building the foundations. Mojo is also a work in
progress, so feel free to contribute and identify bugs and errors.

# Docs

## Tensor

The tensor is the basic building block of deep learning. It is a generalization of vectors and matrices to potentially higher dimensions.
In this repo, tensors consist of a `DynamicVector` that holds data, where data is stored as a `SIMD` data structure. This allows faster calculation. For now, we do not deal with high number of features.

Creating a tensor should be pretty simple once you have your SIMD data in a DynamicVector. For example:

```python
from tensor import Tensor

a = SIMD[DType.float32, 2](3.0, 2.0) # [3.0, 2.0]
d = SIMD[DType.float32, 2](2.0, 3.0) # [2.0, 3.0]

b = DynamicVector[SIMD[DType.float32, 2]](1)
c = DynamicVector[SIMD[DType.float32, 2]](1)

b.push_back(a) # [[3.0, 2.0]]
c.push_back(c) # [[2.0, 3.0]]

let t1 : Tensor = Tensor(b, String("Tensor_name"))
print(t1.name) # Tensor_name
print(t1.size) # 1 (number of SIMDs or dimensions)
print(t1.tensor[0]) # [3.0, 2.0]

let t2 : Tensor = Tensor(c, String("tensor2"))
```

And for now, we can to operations such as:
- tensor sum: `t1 + t2 # [5.0, 5.0]`
- tensor subtraction: `t1 - t2 # [1.0, -1.0]`
- tensor multiplication: `t1 * t2 # [6.0, 6.0]`
- tensor multiplied by float: `t1 * 2.0 # [6.0, 4.0]`
- tensor division: `t1 / t2 # [1.5, 0.6666667]`
- tensor dot product: `t1.dot(t2) # 12.0`
- tensor int power: `t1.power(2) # [9.0, 4.0]`


## Algorithms

### Gradient Descent

Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point. Implemented in `gradient_descent.mojo`, we can use it as following:

```python
at = SIMD[DType.float32, 2](3.0, 2.0)
dt = SIMD[DType.float32, 2](5.0, 4.0)
bt = DynamicVector[SIMD[DType.float32, 2]](2)
bt.push_back(at)
bt.push_back(dt)

ct = DynamicVector[SIMD[DType.float32, 2]](1)
ct.push_back(dt)

let jt = Tensor(bt, String("alo"))
let mt = Tensor(ct, String("de"))

var weights = SIMD[DType.float32, 2](1.0, 1.0)
var weightsV = DynamicVector[SIMD[DType.float32, 2]](2)
weightsV.push_back(weights)
weightsV.push_back(weights)
var wt = Tensor(weightsV, String("a"))

let resultt = gradient_descent(wt, jt, mt, 0.01, 100)

print(resultt.name)
print(resultt.tensor[0])
print(resultt.tensor[1])

# a
# [173823651479552.0, 7.0064923216240854e-45]
# [0.0, 0.0]
```

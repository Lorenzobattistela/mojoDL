from tensor import Tensor

fn relu(x: Tensor) -> Tensor:
    var r = DynamicVector[SIMD[DType.float32, 2]](x.size)
    let y = SIMD[DType.float32, 1](0)
    for i in range(x.size):
        r.push_back(x.tensor[i].max(y))
    let res = Tensor(r, x.name)
    return res
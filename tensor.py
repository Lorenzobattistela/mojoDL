from Vector import DynamicVector
from SIMD import SIMD
from DType import DType
from String import String

struct Tensor:
    var tensor : DynamicVector[SIMD[DType.float32, 2]];
    var name : String;
    var size : Int;

    fn __init__(inout self, tensor: DynamicVector[SIMD[DType.float32, 2]], name: String) -> None:
        self.tensor = tensor;
        self.name = name;
        self.size = len(tensor);
    
    fn __copyinit__(inout self, existing: Tensor) -> None:
        self.tensor = existing.tensor
        self.name = existing.name
        self.size = existing.size
    
    fn __add__(self, other: Tensor) -> Tensor:
        var new_tensor = DynamicVector[SIMD[DType.float32, 2]](self.size)
        for i in range(self.size):
            new_tensor[i] = self.tensor[i] + other.tensor[i]
        let result = Tensor(new_tensor, self.name)
        return result

    fn __mul__(self, other: Tensor) -> Tensor:
        var new_tensor = DynamicVector[SIMD[DType.float32, 2]](self.size)
        for i in range(self.size):
            new_tensor[i] = self.tensor[i] * other.tensor[i]
        let result = Tensor(new_tensor, self.name)
        return result


# a = SIMD[DType.float32, 2](3.0, 2.0)
# d = SIMD[DType.float32, 2](3.0, 2.0)
# b = DynamicVector[SIMD[DType.float32, 2]](1)
# b.push_back(a)


# c = DynamicVector[SIMD[DType.float32, 2]](1)
# c.push_back(d)

# let j = Tensor(b, String("alo"))
# let m = Tensor(c, String("de"))

# let n : Tensor = j + m

# let h : Tensor = j * m

# print(h.tensor[0][0])
# print(n.tensor[0][0])
# 9.0
# 6.0
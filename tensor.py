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
    
    fn printTensor(tensor: Tensor) -> None:
        print(tensor.name)
        for i in range(tensor.size):
            print(tensor.tensor[i])
    
    fn power(self, power: Int) -> Tensor:
        print(self.size)
        var new_new_t = DynamicVector[SIMD[DType.float32, 2]](self.size)
        for i in range(power):
            var new_tensor = DynamicVector[SIMD[DType.float32, 2]](self.size)
            for j in range(self.size):
                new_tensor[j] = self.tensor[j] * self.tensor[j]
            new_new_t = new_tensor
        print(new_new_t[0])
        print(len(new_new_t))
        let result = Tensor(new_new_t, self.name)
        return result
    
    fn __add__(self, other: Tensor) -> Tensor:
        var new_tensor = DynamicVector[SIMD[DType.float32, 2]](self.size)
        for i in range(self.size):
            new_tensor[i] = self.tensor[i] + other.tensor[i]
        let result = Tensor(new_tensor, self.name)
        return result
    
    fn __sub__(self, other: Tensor) -> Tensor:
        var new_tensor = DynamicVector[SIMD[DType.float32, 2]](self.size)
        for i in range(self.size):
            new_tensor[i] = self.tensor[i] - other.tensor[i]
        let result = Tensor(new_tensor, self.name)
        return result

    fn __mul__(self, other: Tensor) -> Tensor:
        var new_tensor = DynamicVector[SIMD[DType.float32, 2]](self.size)
        for i in range(self.size):
            new_tensor[i] = self.tensor[i] * other.tensor[i]
        let result = Tensor(new_tensor, self.name)
        return result
    
    fn __mul__(self, n: Float32) -> Tensor:
        var new_tensor = DynamicVector[SIMD[DType.float32, 2]](self.size)
        for i in range(self.size):
            new_tensor[i] = self.tensor[i] * SIMD[DType.float32, 1](n)
        let result = Tensor(new_tensor, self.name)
        return result
    
    fn __truediv__(self, other: Tensor) -> Tensor:
        var new_tensor = DynamicVector[SIMD[DType.float32, 2]](self.size)
        for i in range(self.size):
            new_tensor[i] = self.tensor[i] / other.tensor[i]
        let result = Tensor(new_tensor, self.name)
        return result
    
    fn dot(self, other: Tensor) -> DynamicVector[Float32]:
        var r = DynamicVector[Float32](self.size)
        for i in range(self.size):
            let summed = self * other
            r.push_back(summed.tensor[i].reduce_add())
        return r


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
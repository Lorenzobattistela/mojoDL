from tensor import Tensor

fn gradient_descent(inout w: Tensor, x: Tensor, y: Tensor, learning_rate: Float32, epochs: Int) -> Tensor:
    for _ in range(epochs):
        # Calculate the loss
        let loss = (y - w * x).dot(y - w * x)

        # Calculate the gradient
        let gradient = (y - w * x) * x

        # Update the weights
        w = w - (gradient * learning_rate)

    return w

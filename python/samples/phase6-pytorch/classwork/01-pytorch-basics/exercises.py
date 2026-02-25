"""
PyTorch Basics Exercises
=========================
Practice problems for PyTorch fundamentals.

Topics Covered:
- Tensor creation and manipulation
- Basic operations
- NumPy conversion
- GPU acceleration
- Device management

Run: python exercises.py
"""

import torch
import numpy as np
import time

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def exercise_1():
    """
    Exercise 1: Tensor Creation
    Create the following tensors:
    a) A 3x3 tensor of zeros
    b) A 2x4 tensor of ones
    c) A 5x5 identity matrix
    d) A 3x3 tensor with random values from normal distribution
    e) A 1D tensor with values from 0 to 9
    """
    print("\n" + "=" * 60)
    print("Exercise 1: Tensor Creation")
    print("=" * 60)

    # Your code here
    a = torch.zeros(3, 3)
    b = torch.ones(2, 4)
    c = torch.eye(5)
    d = torch.randn(3, 3)
    e = torch.arange(10)

    print(f"\na) 3x3 zeros:\n{a}")
    print(f"\nb) 2x4 ones:\n{b}")
    print(f"\nc) 5x5 identity:\n{c}")
    print(f"\nd) 3x3 random normal:\n{d}")
    print(f"\ne) Range 0-9: {e}")

    # Verification
    assert a.shape == (3, 3) and torch.all(a == 0), "Error in a"
    assert b.shape == (2, 4) and torch.all(b == 1), "Error in b"
    assert c.shape == (5, 5) and torch.all(c.diagonal() == 1), "Error in c"
    assert d.shape == (3, 3), "Error in d"
    assert torch.all(e == torch.arange(10)), "Error in e"
    print("\n✓ All assertions passed!")


def exercise_2():
    """
    Exercise 2: Tensor Operations
    Given two tensors:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])

    Compute:
    a) Element-wise addition
    b) Element-wise multiplication
    c) Sum of all elements in x
    d) Mean of x
    e) Standard deviation of x
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Tensor Operations")
    print("=" * 60)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])

    print(f"\nx = {x}")
    print(f"y = {y}")

    # Your code here
    a = x + y
    b = x * y
    c = torch.sum(x)
    d = torch.mean(x)
    e = torch.std(x)

    print(f"\na) x + y = {a}")
    print(f"b) x * y = {b}")
    print(f"c) sum(x) = {c.item()}")
    print(f"d) mean(x) = {d.item()}")
    print(f"e) std(x) = {e.item():.4f}")

    # Verification
    assert torch.allclose(a, torch.tensor([3., 4., 5., 6., 7.])), "Error in a"
    assert torch.allclose(b, torch.tensor([2., 4., 6., 8., 10.])), "Error in b"
    assert c.item() == 15.0, "Error in c"
    assert d.item() == 3.0, "Error in d"
    print("\n✓ All assertions passed!")


def exercise_3():
    """
    Exercise 3: Matrix Operations
    Given matrices:
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]

    Compute:
    a) Matrix multiplication A @ B
    b) Element-wise multiplication A * B
    c) Transpose of A
    d) Determinant of A (use torch.det)
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Matrix Operations")
    print("=" * 60)

    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    print(f"\nA = \n{A}")
    print(f"\nB = \n{B}")

    # Your code here
    a = A @ B  # or torch.matmul(A, B)
    b = A * B
    c = A.T
    d = torch.det(A)

    print(f"\na) A @ B = \n{a}")
    print(f"\nb) A * B (element-wise) = \n{b}")
    print(f"\nc) A.T = \n{c}")
    print(f"\nd) det(A) = {d.item()}")

    # Verification
    expected_matmul = torch.tensor([[19., 22.], [43., 50.]])
    assert torch.allclose(a, expected_matmul), "Error in a"
    assert torch.allclose(b, A * B), "Error in b"
    assert torch.allclose(c, A.T), "Error in c"
    assert abs(d.item() - (-2.0)) < 0.001, "Error in d"
    print("\n✓ All assertions passed!")


def exercise_4():
    """
    Exercise 4: Reshaping and Indexing
    Given tensor:
    x = torch.arange(24)

    a) Reshape it to (4, 6)
    b) Reshape it to (2, 3, 4)
    c) Extract the first row from reshaped (4, 6)
    d) Extract the last column from reshaped (4, 6)
    e) Flatten it back to 1D
    """
    print("\n" + "=" * 60)
    print("Exercise 4: Reshaping and Indexing")
    print("=" * 60)

    x = torch.arange(24)
    print(f"\nOriginal x: {x}")

    # Your code here
    a = x.reshape(4, 6)
    b = x.reshape(2, 3, 4)
    c = a[0]  # or a[0, :]
    d = a[:, -1]
    e = a.flatten()  # or a.reshape(-1)

    print(f"\na) Reshape to (4, 6):\n{a}")
    print(f"\nb) Reshape to (2, 3, 4) shape: {b.shape}")
    print(f"\nc) First row: {c}")
    print(f"\nd) Last column: {d}")
    print(f"\ne) Flattened: shape {e.shape}")

    # Verification
    assert a.shape == (4, 6), "Error in a"
    assert b.shape == (2, 3, 4), "Error in b"
    assert torch.all(c == torch.arange(6)), "Error in c"
    assert torch.all(d == torch.tensor([5, 11, 17, 23])), "Error in d"
    assert e.shape == (24,), "Error in e"
    print("\n✓ All assertions passed!")


def exercise_5():
    """
    Exercise 5: NumPy Conversion
    a) Create a NumPy array: np_arr = np.array([[1, 2, 3], [4, 5, 6]])
    b) Convert it to PyTorch tensor
    c) Multiply the tensor by 2
    d) Convert back to NumPy
    e) Verify the result
    """
    print("\n" + "=" * 60)
    print("Exercise 5: NumPy Conversion")
    print("=" * 60)

    # Your code here
    np_arr = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"\na) NumPy array:\n{np_arr}")

    torch_tensor = torch.from_numpy(np_arr)
    print(f"\nb) PyTorch tensor:\n{torch_tensor}")

    torch_tensor_x2 = torch_tensor * 2
    print(f"\nc) Multiplied by 2:\n{torch_tensor_x2}")

    np_result = torch_tensor_x2.numpy()
    print(f"\nd) Back to NumPy:\n{np_result}")

    expected = np.array([[2, 4, 6], [8, 10, 12]])
    print(f"\ne) Expected:\n{expected}")

    # Verification
    assert np.array_equal(np_result, expected), "Error in conversion"
    print("\n✓ All assertions passed!")


def exercise_6():
    """
    Exercise 6: GPU Operations (if available)
    a) Create a tensor on CPU
    b) Move it to GPU (if available)
    c) Perform an operation on GPU
    d) Move result back to CPU
    e) Verify device at each step
    """
    print("\n" + "=" * 60)
    print("Exercise 6: GPU Operations")
    print("=" * 60)

    # Your code here
    x_cpu = torch.randn(3, 3)
    print(f"\na) Created on CPU:")
    print(f"   Device: {x_cpu.device}")
    print(f"   Tensor:\n{x_cpu}")

    if torch.cuda.is_available():
        x_gpu = x_cpu.to('cuda')
        print(f"\nb) Moved to GPU:")
        print(f"   Device: {x_gpu.device}")

        y_gpu = x_gpu * 2 + 1
        print(f"\nc) Operation on GPU:")
        print(f"   Device: {y_gpu.device}")

        y_cpu = y_gpu.cpu()
        print(f"\nd) Moved back to CPU:")
        print(f"   Device: {y_cpu.device}")
        print(f"   Result:\n{y_cpu}")

        # Verification
        assert x_cpu.device.type == 'cpu', "Error: x_cpu not on CPU"
        assert x_gpu.device.type == 'cuda', "Error: x_gpu not on GPU"
        assert y_gpu.device.type == 'cuda', "Error: y_gpu not on GPU"
        assert y_cpu.device.type == 'cpu', "Error: y_cpu not on CPU"
        assert torch.allclose(y_cpu, x_cpu * 2 + 1), "Error in computation"
        print("\n✓ All assertions passed!")
    else:
        print("\nb) GPU not available - skipping GPU tests")
        print("   Code structure would be:")
        print("   x_gpu = x_cpu.to('cuda')")
        print("   y_gpu = x_gpu * 2 + 1")
        print("   y_cpu = y_gpu.cpu()")


def exercise_7():
    """
    Exercise 7: Broadcasting
    a) Create tensor a with shape (3, 1) with values [[1], [2], [3]]
    b) Create tensor b with shape (1, 4) with values [[10, 20, 30, 40]]
    c) Add them together (broadcasting)
    d) Verify the result shape is (3, 4)
    e) Print the result
    """
    print("\n" + "=" * 60)
    print("Exercise 7: Broadcasting")
    print("=" * 60)

    # Your code here
    a = torch.tensor([[1.0], [2.0], [3.0]])
    b = torch.tensor([[10.0, 20.0, 30.0, 40.0]])

    print(f"\na) Tensor a (3, 1):\n{a}")
    print(f"\nb) Tensor b (1, 4):\n{b}")

    c = a + b
    print(f"\nc) a + b (broadcasted):\n{c}")
    print(f"   Shape: {c.shape}")

    expected = torch.tensor([
        [11., 21., 31., 41.],
        [12., 22., 32., 42.],
        [13., 23., 33., 43.]
    ])

    # Verification
    assert c.shape == (3, 4), "Error: wrong shape"
    assert torch.allclose(c, expected), "Error in broadcasting result"
    print("\n✓ All assertions passed!")


def exercise_8():
    """
    Exercise 8: Performance Comparison
    Compare the time it takes to:
    a) Matrix multiply two 1000x1000 matrices on CPU
    b) Matrix multiply two 1000x1000 matrices on GPU (if available)
    c) Calculate and display the speedup
    """
    print("\n" + "=" * 60)
    print("Exercise 8: Performance Comparison")
    print("=" * 60)

    size = 1000
    iterations = 10

    # CPU benchmark
    print(f"\nBenchmarking {size}x{size} matrix multiplication ({iterations} iterations)...")

    A_cpu = torch.randn(size, size)
    B_cpu = torch.randn(size, size)

    start = time.time()
    for _ in range(iterations):
        C_cpu = torch.matmul(A_cpu, B_cpu)
    cpu_time = time.time() - start

    print(f"\na) CPU time: {cpu_time:.4f} seconds")

    # GPU benchmark (if available)
    if torch.cuda.is_available():
        A_gpu = A_cpu.cuda()
        B_gpu = B_cpu.cuda()

        # Warm up
        for _ in range(5):
            _ = torch.matmul(A_gpu, B_gpu)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(iterations):
            C_gpu = torch.matmul(A_gpu, B_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        print(f"b) GPU time: {gpu_time:.4f} seconds")

        speedup = cpu_time / gpu_time
        print(f"\nc) Speedup: {speedup:.2f}x")

        if speedup > 1:
            print(f"   GPU is {speedup:.2f}x faster!")
        else:
            print(f"   CPU is {1/speedup:.2f}x faster (overhead for small ops)")
    else:
        print(f"\nb) GPU not available - skipping GPU benchmark")
        print("   With CUDA GPU, you would see significant speedup!")


def bonus_exercise():
    """
    Bonus Exercise: Create a Simple Linear Model
    Implement a simple linear regression: y = Wx + b
    - Create random weight W and bias b
    - Generate random input data X
    - Compute predictions y = X @ W + b
    - Compute mean squared error loss
    """
    print("\n" + "=" * 60)
    print("Bonus Exercise: Simple Linear Model")
    print("=" * 60)

    # Your code here
    # Data: 5 samples, 3 features
    X = torch.randn(5, 3)
    # True targets
    y_true = torch.randn(5, 1)

    print(f"\nInput X (5 samples, 3 features):\n{X}")
    print(f"\nTrue targets y:\n{y_true}")

    # Model parameters
    W = torch.randn(3, 1)
    b = torch.randn(1)

    print(f"\nWeight W:\n{W}")
    print(f"Bias b: {b}")

    # Forward pass
    y_pred = X @ W + b

    print(f"\nPredictions:\n{y_pred}")

    # Loss (Mean Squared Error)
    loss = torch.mean((y_pred - y_true) ** 2)

    print(f"\nMean Squared Error Loss: {loss.item():.4f}")

    # Verification
    assert y_pred.shape == (5, 1), "Error: wrong prediction shape"
    assert loss.ndim == 0, "Error: loss should be scalar"
    print("\n✓ All assertions passed!")
    print("\nThis is the foundation of neural networks!")


def main():
    """Run all exercises."""
    print("=" * 60)
    print("PyTorch Basics Exercises")
    print("=" * 60)
    print(f"\nDevice: {device}")
    print("Work through each exercise to practice PyTorch fundamentals.")

    try:
        exercise_1()
        exercise_2()
        exercise_3()
        exercise_4()
        exercise_5()
        exercise_6()
        exercise_7()
        exercise_8()
        bonus_exercise()

        print("\n" + "=" * 60)
        print("🎉 Congratulations! All exercises completed!")
        print("=" * 60)
        print("\nYou've mastered:")
        print("  ✓ Tensor creation and manipulation")
        print("  ✓ Basic tensor operations")
        print("  ✓ Matrix operations")
        print("  ✓ Reshaping and indexing")
        print("  ✓ NumPy conversion")
        print("  ✓ GPU acceleration")
        print("  ✓ Broadcasting")
        print("  ✓ Performance optimization")
        print("\nNext steps:")
        print("  • Move on to 02-tensors-operations/ for advanced topics")
        print("  • Experiment with your own tensor operations")
        print("  • Try building simple mathematical models")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Review the exercise and try again.")


if __name__ == "__main__":
    main()

"""
PyTorch Installation and Setup
==============================
Verify PyTorch installation and check GPU availability.

Topics:
- Installation verification
- GPU/CUDA availability
- PyTorch version info
- Basic configuration
"""

import sys

def check_pytorch():
    """Check if PyTorch is installed and display version info."""
    try:
        import torch
        print("✓ PyTorch is installed")
        print(f"  Version: {torch.__version__}")
        return torch
    except ImportError:
        print("✗ PyTorch is NOT installed")
        print("\nTo install PyTorch:")
        print("  CPU only: pip install torch torchvision torchaudio")
        print("  GPU (CUDA 11.8): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("  GPU (CUDA 12.1): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\nOr visit: https://pytorch.org/get-started/locally/")
        return None

def check_cuda(torch):
    """Check CUDA availability and GPU information."""
    if torch is None:
        return
    
    print("\nCUDA Information:")
    print("-" * 40)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("No GPU available. PyTorch will use CPU.")
        print("\nTo enable GPU:")
        print("1. Install NVIDIA GPU driver")
        print("2. Install CUDA toolkit")
        print("3. Install PyTorch with CUDA support")
        print("4. OR use Google Colab for free GPU access")

def check_dependencies():
    """Check other important dependencies."""
    print("\nOther Dependencies:")
    print("-" * 40)
    
    dependencies = {
        'numpy': 'NumPy',
        'torchvision': 'TorchVision',
        'torchaudio': 'TorchAudio',
        'matplotlib': 'Matplotlib',
        'tensorboard': 'TensorBoard',
    }
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (not required, but recommended)")

def test_basic_operation(torch):
    """Test basic PyTorch operations."""
    if torch is None:
        return
    
    print("\nBasic Operation Test:")
    print("-" * 40)
    
    # Create tensors
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x + y = {x + y}")
    print(f"x * y = {x * y}")
    
    # Matrix multiplication
    A = torch.randn(3, 4)
    B = torch.randn(4, 2)
    C = torch.matmul(A, B)
    
    print(f"\nMatrix multiplication:")
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print(f"C = A @ B shape: {C.shape}")
    
    # GPU test (if available)
    if torch.cuda.is_available():
        print("\nGPU Operation Test:")
        x_gpu = x.cuda()
        y_gpu = y.cuda()
        z_gpu = x_gpu + y_gpu
        print(f"x (GPU) + y (GPU) = {z_gpu}")
        print(f"Result device: {z_gpu.device}")

def print_system_info():
    """Print Python and system information."""
    print("\nSystem Information:")
    print("-" * 40)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Python Path: {sys.executable}")

def main():
    print("=" * 60)
    print("PyTorch Installation and Setup Check")
    print("=" * 60)
    
    # Check PyTorch
    torch = check_pytorch()
    
    # Check CUDA
    check_cuda(torch)
    
    # Check dependencies
    check_dependencies()
    
    # Test basic operations
    test_basic_operation(torch)
    
    # System info
    print_system_info()
    
    print("\n" + "=" * 60)
    if torch and torch.cuda.is_available():
        print("✓ All set! You're ready to use PyTorch with GPU acceleration.")
    elif torch:
        print("✓ PyTorch is installed. You can start learning (CPU mode).")
    else:
        print("Please install PyTorch before continuing.")
    print("=" * 60)

if __name__ == "__main__":
    main()

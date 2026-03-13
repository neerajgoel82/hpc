"""
ONNX Export
Export PyTorch models to ONNX format for deployment

This module demonstrates:
- Exporting models to ONNX
- ONNX model verification
- Running inference with ONNX Runtime
- Dynamic vs static shapes
- Model optimization for deployment

Note: Install with: pip install onnx onnxruntime
Run: python 06_onnx_export.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import onnx
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX not installed. Install with: pip install onnx onnxruntime")


class SimpleClassifier(nn.Module):
    """Simple classifier for ONNX export."""

    def __init__(self, input_dim=100, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ConvNet(nn.Module):
    """Convolutional network for image classification."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def demonstrate_basic_export():
    """Demonstrate basic ONNX export."""
    print("1. Basic ONNX Export")
    print("-" * 60)

    # Create model
    model = SimpleClassifier(input_dim=100, num_classes=10)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 100)

    # Export to ONNX
    onnx_path = "/tmp/simple_classifier.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Model exported to: {onnx_path}")

    if ONNX_AVAILABLE:
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid")
        print(f"ONNX version: {onnx.__version__}")

    print("\nExport parameters:")
    print("  - export_params=True: Include trained weights")
    print("  - opset_version: ONNX operator set version")
    print("  - do_constant_folding: Optimize constants")
    print("  - dynamic_axes: Support variable batch size")
    print()


def demonstrate_inference():
    """Demonstrate ONNX inference."""
    print("2. ONNX Runtime Inference")
    print("-" * 60)

    if not ONNX_AVAILABLE:
        print("ONNX Runtime not available.\n")
        return

    # Create and export model
    model = SimpleClassifier(input_dim=100, num_classes=10)
    model.eval()

    dummy_input = torch.randn(1, 100)
    onnx_path = "/tmp/classifier_inference.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}},
    )

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = model(dummy_input).numpy()

    # ONNX Runtime inference
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # Compare outputs
    difference = np.abs(pytorch_output - ort_output).max()
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"ONNX output shape: {ort_output.shape}")
    print(f"Max difference: {difference:.8f}")
    print(f"Outputs match: {np.allclose(pytorch_output, ort_output, rtol=1e-5)}")
    print()


def demonstrate_dynamic_shapes():
    """Demonstrate dynamic vs static shapes."""
    print("3. Dynamic vs Static Shapes")
    print("-" * 60)

    model = SimpleClassifier()
    model.eval()

    # Static shape export
    dummy_input = torch.randn(4, 100)
    static_path = "/tmp/static_shape.onnx"

    torch.onnx.export(
        model, dummy_input, static_path, input_names=["input"], output_names=["output"]
    )

    print("Static shape model:")
    print("  - Fixed batch size (4)")
    print("  - Cannot handle different batch sizes")
    print("  - Slightly faster inference")

    # Dynamic shape export
    dynamic_path = "/tmp/dynamic_shape.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        dynamic_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print("\nDynamic shape model:")
    print("  - Variable batch size")
    print("  - Flexible for deployment")
    print("  - Slightly slower (minimal)")

    if ONNX_AVAILABLE:
        # Test dynamic shapes
        ort_session = ort.InferenceSession(dynamic_path)

        for batch_size in [1, 2, 8]:
            test_input = torch.randn(batch_size, 100).numpy()
            ort_inputs = {ort_session.get_inputs()[0].name: test_input}
            output = ort_session.run(None, ort_inputs)[0]
            print(f"  Batch size {batch_size}: Output shape {output.shape}")

    print()


def demonstrate_convnet_export():
    """Demonstrate CNN export."""
    print("4. Convolutional Network Export")
    print("-" * 60)

    model = ConvNet(num_classes=10)
    model.eval()

    # Export with image input
    dummy_input = torch.randn(1, 3, 32, 32)
    onnx_path = "/tmp/convnet.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    print(f"ConvNet exported to: {onnx_path}")
    print(f"Input shape: {dummy_input.shape} (batch, channels, height, width)")

    if ONNX_AVAILABLE:
        # Verify
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # Inference
        ort_session = ort.InferenceSession(onnx_path)
        test_images = torch.randn(4, 3, 32, 32).numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: test_images}
        output = ort_session.run(None, ort_inputs)[0]

        print(f"Test batch shape: {test_images.shape}")
        print(f"Output shape: {output.shape}")

    print()


def demonstrate_best_practices():
    """Demonstrate ONNX export best practices."""
    print("5. Best Practices")
    print("-" * 60)

    print("ONNX Export Best Practices:")
    print()
    print("1. Model Preparation:")
    print("   - Set model to eval mode: model.eval()")
    print("   - Use representative dummy input")
    print("   - Handle custom operations carefully")
    print()
    print("2. Export Configuration:")
    print("   - Use appropriate opset_version (11+)")
    print("   - Enable constant folding")
    print("   - Define meaningful input/output names")
    print("   - Use dynamic axes for flexibility")
    print()
    print("3. Verification:")
    print("   - Check model with onnx.checker")
    print("   - Compare PyTorch vs ONNX outputs")
    print("   - Test with different input shapes")
    print()
    print("4. Optimization:")
    print("   - Use ONNX optimizers")
    print("   - Quantize for mobile/edge devices")
    print("   - Profile inference performance")
    print()
    print("5. Deployment:")
    print("   - ONNX Runtime for production")
    print("   - TensorRT for NVIDIA GPUs")
    print("   - CoreML for iOS devices")
    print("   - OpenVINO for Intel hardware")
    print()


def demonstrate_common_issues():
    """Demonstrate common ONNX export issues."""
    print("6. Common Issues and Solutions")
    print("-" * 60)

    print("Common ONNX Export Issues:")
    print()
    print("1. Unsupported Operations:")
    print("   Problem: Some PyTorch ops not in ONNX")
    print("   Solution: Use ONNX-compatible alternatives")
    print()
    print("2. Dynamic Control Flow:")
    print("   Problem: If/else based on tensor values")
    print("   Solution: Avoid or use scripting")
    print()
    print("3. In-place Operations:")
    print("   Problem: Operations like x += 1")
    print("   Solution: Use x = x + 1 instead")
    print()
    print("4. Custom Layers:")
    print("   Problem: Custom nn.Module not exportable")
    print("   Solution: Register symbolic function")
    print()
    print("5. Shape Inference:")
    print("   Problem: Cannot infer output shape")
    print("   Solution: Provide clearer operations")
    print()


def demonstrate_model_info():
    """Demonstrate extracting ONNX model information."""
    print("7. Model Information")
    print("-" * 60)

    if not ONNX_AVAILABLE:
        print("ONNX not available.\n")
        return

    # Export simple model
    model = SimpleClassifier(input_dim=100, num_classes=10)
    model.eval()
    dummy_input = torch.randn(1, 100)
    onnx_path = "/tmp/model_info.onnx"

    torch.onnx.export(model, dummy_input, onnx_path)

    # Load and inspect
    onnx_model = onnx.load(onnx_path)

    print("Model Information:")
    print(f"  IR version: {onnx_model.ir_version}")
    print(f"  Producer: {onnx_model.producer_name}")
    print(f"  Graph name: {onnx_model.graph.name}")
    print(f"  Inputs: {len(onnx_model.graph.input)}")
    print(f"  Outputs: {len(onnx_model.graph.output)}")
    print(f"  Nodes: {len(onnx_model.graph.node)}")

    print("\nInput details:")
    for input_tensor in onnx_model.graph.input:
        print(f"  Name: {input_tensor.name}")

    print("\nOutput details:")
    for output_tensor in onnx_model.graph.output:
        print(f"  Name: {output_tensor.name}")

    print()


def main():
    print("=" * 70)
    print("ONNX MODEL EXPORT")
    print("=" * 70)
    print()

    print(f"PyTorch version: {torch.__version__}")
    if ONNX_AVAILABLE:
        print(f"ONNX version: {onnx.__version__}")
        print(f"ONNX Runtime version: {ort.__version__}")
    print()

    if not ONNX_AVAILABLE:
        print("Install ONNX and ONNX Runtime:")
        print("  pip install onnx onnxruntime")
        print()

    demonstrate_basic_export()
    demonstrate_inference()
    demonstrate_dynamic_shapes()
    demonstrate_convnet_export()
    demonstrate_best_practices()
    demonstrate_common_issues()
    demonstrate_model_info()

    print("=" * 70)


if __name__ == "__main__":
    main()

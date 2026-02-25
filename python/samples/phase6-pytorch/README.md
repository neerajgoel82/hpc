# Phase 6: Deep Learning with PyTorch

**Duration**: 8-10 weeks
**Level**: Advanced
**Prerequisites**: Phase 5 (Data Science), Basic Linear Algebra, Calculus (helpful)

---

## Overview

Phase 6 introduces deep learning with PyTorch, covering neural networks, computer vision, sequence models, and production deployment. You'll learn industry-standard deep learning techniques used at companies like Meta, Tesla, and OpenAI.

---

## Learning Objectives

By the end of this phase, you will:
- ✅ Master PyTorch tensor operations
- ✅ Understand automatic differentiation (autograd)
- ✅ Build and train neural networks from scratch
- ✅ Implement CNNs for image processing
- ✅ Create RNNs/LSTMs for sequence data
- ✅ Apply transfer learning with pretrained models
- ✅ Deploy models in production
- ✅ Connect Python with CUDA kernels (HPC link)

---

## Module Structure

### 01. PyTorch Basics (Week 1)
**Directory**: `01-pytorch-basics/`

**Topics**:
- PyTorch installation and ecosystem
- Tensors vs NumPy arrays
- GPU acceleration basics
- PyTorch workflow overview
- Debugging and best practices

**Files**:
- `01_installation_setup.py` - Environment setup
- `02_pytorch_intro.py` - First PyTorch program
- `03_tensor_basics.py` - Creating and manipulating tensors
- `04_numpy_vs_pytorch.py` - Comparison and conversion
- `05_gpu_acceleration.py` - Using CUDA with PyTorch
- `exercises.py` - Practice problems

**Key Concepts**: Tensor operations, GPU acceleration, PyTorch ecosystem

---

### 02. Tensors & Operations (Week 1-2)
**Directory**: `02-tensors-operations/`

**Topics**:
- Tensor creation and initialization
- Indexing, slicing, and reshaping
- Mathematical operations
- Broadcasting
- In-place vs out-of-place operations
- Memory management

**Files**:
- `01_tensor_creation.py` - Creating tensors
- `02_indexing_slicing.py` - Tensor manipulation
- `03_math_operations.py` - Element-wise and matrix ops
- `04_broadcasting.py` - Broadcasting rules
- `05_reshaping_views.py` - Reshape, view, transpose
- `06_memory_management.py` - Efficient memory use
- `exercises.py` - Tensor manipulation practice

**Key Concepts**: Tensor manipulation, broadcasting, views vs copies

---

### 03. Autograd & Backpropagation (Week 2)
**Directory**: `03-autograd-backprop/`

**Topics**:
- Computational graphs
- Automatic differentiation
- Gradient computation
- `requires_grad` and grad_fn
- Custom autograd functions
- Gradient accumulation

**Files**:
- `01_autograd_basics.py` - Automatic differentiation
- `02_computational_graphs.py` - Understanding graphs
- `03_gradient_computation.py` - Computing gradients
- `04_custom_autograd.py` - Custom backward functions
- `05_gradient_descent.py` - Manual optimization
- `06_gradient_accumulation.py` - Gradient accumulation
- `exercises.py` - Autograd practice

**Key Concepts**: Backpropagation, computational graphs, gradient descent

---

### 04. Neural Networks (Week 3-4)
**Directory**: `04-neural-networks/`

**Topics**:
- torch.nn.Module
- Linear layers
- Activation functions
- Loss functions
- Optimizers (SGD, Adam, etc.)
- Training loops
- Validation and testing

**Files**:
- `01_nn_module.py` - Creating custom modules
- `02_linear_layers.py` - Fully connected layers
- `03_activations.py` - ReLU, Sigmoid, Tanh, etc.
- `04_loss_functions.py` - MSE, CrossEntropy, etc.
- `05_optimizers.py` - SGD, Adam, AdamW
- `06_training_loop.py` - Complete training pipeline
- `07_mlp_classification.py` - Multi-layer perceptron
- `08_regularization.py` - Dropout, weight decay
- `exercises.py` - Build and train networks

**Key Concepts**: Neural network architecture, training loop, optimization

---

### 05. Training & Optimization (Week 4)
**Directory**: `05-training-optimization/`

**Topics**:
- Learning rate scheduling
- Batch normalization
- Data augmentation
- Early stopping
- Checkpointing
- TensorBoard integration
- Hyperparameter tuning

**Files**:
- `01_learning_rate_scheduling.py` - LR schedulers
- `02_batch_normalization.py` - Batch norm layers
- `03_data_augmentation.py` - Augmentation techniques
- `04_early_stopping.py` - Stopping criteria
- `05_checkpointing.py` - Saving/loading models
- `06_tensorboard.py` - Visualization with TensorBoard
- `07_hyperparameter_tuning.py` - Grid search, random search
- `exercises.py` - Optimization practice

**Key Concepts**: Training tricks, monitoring, hyperparameter optimization

---

### 06. CNNs & Image Processing (Week 5-6)
**Directory**: `06-cnn-image-processing/`

**Topics**:
- Convolutional layers
- Pooling layers
- CNN architectures (LeNet, AlexNet, VGG)
- Image classification
- torchvision datasets
- Data loaders and transforms

**Files**:
- `01_conv_layers.py` - Convolution operations
- `02_pooling_layers.py` - Max pooling, average pooling
- `03_lenet.py` - LeNet architecture
- `04_image_classification.py` - CIFAR-10 classification
- `05_torchvision_datasets.py` - Using built-in datasets
- `06_data_loaders.py` - Efficient data loading
- `07_custom_datasets.py` - Creating custom datasets
- `08_resnet_transfer.py` - Transfer learning with ResNet
- `exercises.py` - Build CNNs

**Key Concepts**: Convolutions, CNN architectures, image classification

---

### 07. RNNs & Sequence Models (Week 7)
**Directory**: `07-rnn-sequences/`

**Topics**:
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRU)
- Sequence-to-sequence models
- Attention mechanism basics
- Text processing with PyTorch

**Files**:
- `01_rnn_basics.py` - Simple RNN
- `02_lstm.py` - LSTM architecture
- `03_gru.py` - GRU architecture
- `04_sequence_classification.py` - Text classification
- `05_time_series.py` - Time series prediction
- `06_seq2seq.py` - Sequence-to-sequence
- `07_attention.py` - Attention mechanism
- `exercises.py` - Build sequence models

**Key Concepts**: Sequential data, LSTM, attention

---

### 08. Transfer Learning (Week 8)
**Directory**: `08-transfer-learning/`

**Topics**:
- Pretrained models
- Fine-tuning strategies
- Feature extraction
- Domain adaptation
- Model zoo
- Practical transfer learning

**Files**:
- `01_pretrained_models.py` - Loading pretrained models
- `02_feature_extraction.py` - Using as feature extractor
- `03_fine_tuning.py` - Fine-tuning strategies
- `04_domain_adaptation.py` - Adapting to new domains
- `05_model_zoo.py` - torchvision.models overview
- `06_practical_transfer.py` - Real-world example
- `exercises.py` - Transfer learning practice

**Key Concepts**: Pretrained models, fine-tuning, transfer learning

---

### 09. Custom Models & Advanced Topics (Week 9)
**Directory**: `09-custom-models/`

**Topics**:
- Custom architectures
- Model ensembles
- Mixed precision training
- Distributed training basics
- PyTorch Lightning
- ONNX export
- Connecting with CUDA (from Phase 6 CUDA course)

**Files**:
- `01_custom_architectures.py` - Building custom models
- `02_model_ensembles.py` - Ensemble techniques
- `03_mixed_precision.py` - AMP (Automatic Mixed Precision)
- `04_distributed_basics.py` - Multi-GPU training
- `05_pytorch_lightning.py` - Using PyTorch Lightning
- `06_onnx_export.py` - Exporting to ONNX
- `07_cuda_integration.py` - Custom CUDA kernels with PyTorch
- `exercises.py` - Advanced topics

**Key Concepts**: Custom models, optimization, production readiness

---

### 10. Projects (Week 10)
**Directory**: `10-projects/`

**Complete Deep Learning Projects**:

1. **Image Classifier**
   - `project_image_classifier.py`
   - Classify custom image dataset
   - Use: CNNs, transfer learning

2. **Object Detection**
   - `project_object_detection.py`
   - Detect objects in images
   - Use: Pretrained models, fine-tuning

3. **Text Sentiment Analysis**
   - `project_sentiment_analysis.py`
   - Classify text sentiment
   - Use: RNN/LSTM, embeddings

4. **Time Series Forecasting**
   - `project_time_series.py`
   - Predict future values
   - Use: LSTM, sequence models

5. **Style Transfer**
   - `project_style_transfer.py`
   - Artistic style transfer
   - Use: CNNs, optimization

6. **Generative Model (GANs)**
   - `project_simple_gan.py`
   - Generate synthetic images
   - Use: GANs basics

---

## Prerequisites

### Knowledge
- Python (Phases 1-4)
- Data Science (Phase 5)
- Linear algebra (matrix operations)
- Calculus (derivatives - helpful)
- Basic statistics

### Hardware
- GPU recommended (not required)
- Google Colab free GPU (T4)
- Local GPU (NVIDIA with CUDA)

### Tools Required
```bash
pip install torch torchvision torchaudio
pip install tensorboard pytorch-lightning
pip install timm  # PyTorch Image Models
```

---

## Learning Path

### Week 1-2: PyTorch Fundamentals
```
Days 1-3: PyTorch basics and tensors
Days 4-7: Tensor operations
Days 8-10: Autograd and backpropagation
Days 11-14: Neural networks basics
```

### Week 3-4: Deep Learning Essentials
```
Week 3: Training loops, optimization
Week 4: Advanced training techniques
```

### Week 5-6: Computer Vision
```
Week 5: CNNs, image classification
Week 6: Transfer learning, advanced CNNs
```

### Week 7: Sequence Models
```
Days 1-3: RNN and LSTM basics
Days 4-5: Sequence tasks
Days 6-7: Attention mechanism
```

### Week 8: Transfer Learning
```
Master pretrained models
Fine-tuning strategies
Domain adaptation
```

### Week 9: Advanced Topics
```
Custom models
Production techniques
CUDA integration
```

### Week 10: Projects
```
Complete 2-3 projects
Build portfolio
```

---

## Installation & Setup

### Local Installation
```bash
# CPU only
pip install torch torchvision torchaudio

# GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify Installation
```python
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
```

### Google Colab Setup
```python
# Colab has PyTorch pre-installed
# Check GPU
!nvidia-smi

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

---

## PyTorch Workflow

### Standard Training Pipeline
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 1. Define model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# 2. Initialize
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training loop
for epoch in range(epochs):
    for inputs, targets in train_loader:
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 4. Evaluation
model.eval()
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        # compute metrics
```

---

## Best Practices

### GPU Usage
```python
# Move model and data to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    # ... training code ...
```

### Model Saving/Loading
```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = Net()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## Common Patterns

### Custom Dataset
```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = CustomDataset(data, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Training Function
```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)
```

---

## Resources

### Official Resources
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Examples](https://github.com/pytorch/examples)

### Books
- "Deep Learning with PyTorch" by Stevens et al.
- "Programming PyTorch for Deep Learning" by Ian Pointer
- "Dive into Deep Learning" (Free online book)

### Courses
- Fast.ai Practical Deep Learning
- Stanford CS230: Deep Learning
- Andrew Ng's Deep Learning Specialization

### Practice
- Kaggle Competitions
- PyTorch Lightning Bolts
- Papers with Code

### Community
- PyTorch Forums
- r/PyTorch
- PyTorch Discord

---

## Connection to HPC/CUDA

This phase connects directly to CUDA programming:
- **PyTorch + CUDA**: Custom CUDA kernels for PyTorch
- **Performance**: Understanding GPU execution in PyTorch
- **Optimization**: Mixed precision, memory management
- **Production**: Deploy optimized models

See `09-custom-models/07_cuda_integration.py` for PyTorch-CUDA integration.

---

## Next Steps

After completing Phase 6:
- **Advanced Deep Learning**: Transformers, GANs, RL
- **Computer Vision**: Object detection, segmentation
- **NLP**: BERT, GPT, transformers
- **Production ML**: MLOps, model deployment
- **Research**: Read papers, implement from scratch
- **Kaggle**: Compete and learn

---

## Tips for Success

1. **Code every day**: Consistency beats intensity
2. **Start simple**: Master basics before advanced topics
3. **Visualize**: Use TensorBoard, plot everything
4. **Read papers**: Stay current with research
5. **Implement from scratch**: Don't just use libraries
6. **Join communities**: Learn from others
7. **GPU access**: Use Colab if no local GPU
8. **Debug systematically**: Check shapes, values, gradients

---

**Ready to start?** Begin with `01-pytorch-basics/01_installation_setup.py` and work through each module! 🔥🧠

# Quick Start - Phase 6: PyTorch

## Setup (10 minutes)

### 1. Create Virtual Environment
```bash
cd python/samples/phase6-pytorch
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install PyTorch

**CPU Only:**
```bash
pip install torch torchvision torchaudio
```

**GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Other Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python 01-pytorch-basics/01_installation_setup.py
```

## Using Google Colab (Free GPU!)

1. Go to https://colab.research.google.com
2. Runtime → Change runtime type → GPU
3. Upload notebook or run code cells
4. PyTorch pre-installed!

## Learning Sequence

### Week 1: Basics
```bash
cd 01-pytorch-basics
python 01_installation_setup.py
python 02_pytorch_intro.py
# ... continue
```

### Week 2: Tensors & Autograd
```bash
cd ../02-tensors-operations
# ...
cd ../03-autograd-backprop
# ...
```

### Continue through all 10 modules

## Quick Examples

### Tensor Creation
```python
import torch

# Create tensor
x = torch.tensor([1.0, 2.0, 3.0])
print(x)

# GPU (if available)
if torch.cuda.is_available():
    x = x.cuda()
    print(f"Tensor on GPU: {x.device}")
```

### Simple Neural Network
```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
x = torch.randn(5, 10)
output = model(x)
print(output.shape)  # torch.Size([5, 1])
```

### Training Loop Template
```python
import torch
import torch.nn as nn
import torch.optim as optim

model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

## GPU vs CPU

```python
import torch

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model and data to device
model = model.to(device)
data = data.to(device)
```

## Troubleshooting

**Issue**: CUDA not available
**Solution**: 
- Check GPU drivers installed
- Verify CUDA toolkit installation
- Or use CPU mode (works fine for learning)
- Or use Google Colab for free GPU

**Issue**: Out of memory on GPU
**Solution**:
- Reduce batch size
- Use `torch.cuda.empty_cache()`
- Use mixed precision training

**Issue**: Slow training
**Solution**:
- Use GPU if available
- Use DataLoader with multiple workers
- Enable cudNN benchmark: `torch.backends.cudnn.benchmark = True`

## Next Steps

1. Complete all modules sequentially
2. Build projects in `10-projects/`
3. Kaggle competitions
4. Read PyTorch tutorials
5. Implement papers from scratch

## Resources

- PyTorch Docs: https://pytorch.org/docs/
- PyTorch Tutorials: https://pytorch.org/tutorials/
- Fast.ai: https://www.fast.ai/
- Papers with Code: https://paperswithcode.com/
- PyTorch Forums: https://discuss.pytorch.org/

Happy Learning! 🔥🧠

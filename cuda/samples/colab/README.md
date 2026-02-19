# CUDA Learning with Google Colab

This directory contains all resources for learning CUDA programming using Google Colab (no local GPU required).

## Quick Start

1. **Read the setup guide**: `SETUP_WITHOUT_LOCAL_GPU.md`
2. **Review the curriculum**: `CUDA_LEARNING_CURRICULUM.md`
3. **Start with Phase 1 notebooks**: Open `notebooks/phase1/` in Google Colab

## Directory Structure

```
colab/
├── README.md                          # This file
├── SETUP_WITHOUT_LOCAL_GPU.md         # How to use Colab for CUDA learning
├── CUDA_LEARNING_CURRICULUM.md        # Complete learning curriculum
│
├── notebooks/                         # Jupyter notebooks for Colab
│   ├── phase1/                       # Foundations (Weeks 1-2)
│   ├── phase2/                       # Memory Management (Weeks 3-4)
│   ├── phase3/                       # Optimization Fundamentals (Weeks 5-6)
│   ├── phase4/                       # Advanced Memory (Weeks 7-8)
│   ├── phase5/                       # Advanced Algorithms (Weeks 9-10)
│   ├── phase6/                       # Streams & Concurrency (Week 11)
│   ├── phase7/                       # Performance Engineering (Weeks 12-13)
│   ├── phase8/                       # Real-World Applications (Weeks 14-15)
│   └── phase9/                       # Advanced Topics (Week 16+)
│
├── projects/                         # Sample CUDA projects
│   ├── 01-hello-world/
│   ├── 02-device-query/
│   ├── 03-vector-add/
│   └── ...
│
└── docs/                            # Additional documentation
    ├── colab-tips.md
    ├── troubleshooting.md
    └── workflows.md
```

## How to Use These Materials

### Method 1: Direct Colab Execution
1. Go to [Google Colab](https://colab.research.google.com)
2. Upload a notebook from `notebooks/`
3. Enable GPU runtime
4. Run cells

### Method 2: GitHub Integration
1. Upload this repo to your GitHub
2. In Colab: File → Open Notebook → GitHub tab
3. Enter your repo URL
4. Open notebooks directly from GitHub

### Method 3: Google Drive Sync
1. Mount Google Drive in Colab
2. Clone this repo to your Drive
3. Edit and run from Drive
4. Changes persist across sessions

## Getting Started Today

### Step 1: Verify GPU Access
Open this notebook in Colab: `notebooks/phase1/00-setup-verification.ipynb`

Or create a new Colab notebook and run:
```python
!nvidia-smi
!nvcc --version
```

### Step 2: Run Hello World
Open: `notebooks/phase1/01-hello-world.ipynb`

### Step 3: Follow the Curriculum
Work through notebooks sequentially, one phase at a time.

## Tips for Success

- **Save your work**: Copy notebooks to Google Drive
- **Session limits**: Colab sessions last 12 hours max
- **GPU availability**: During peak hours, GPU might not be available
- **Download results**: Save important outputs before session ends
- **Use git**: Commit completed work to track progress

## Cost: $0

All materials in this directory can be completed using Google Colab's free tier.

## Ready to Start?

Head to `notebooks/phase1/` and open the first notebook!

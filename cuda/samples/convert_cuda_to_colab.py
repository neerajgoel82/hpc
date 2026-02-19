#!/usr/bin/env python3
"""
CUDA to Colab Notebook Converter

This script converts .cu (CUDA C++) files to Jupyter notebooks compatible with Google Colab.
It scans the cuda-course-fcc repository and generates notebooks in collab-fcc-course.

Usage:
    python convert_cuda_to_colab.py
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


class CUDAToColabConverter:
    """Converts CUDA .cu files to Google Colab-compatible Jupyter notebooks."""

    def __init__(self, source_repo: str, target_dir: str):
        self.source_repo = Path(source_repo)
        self.target_dir = Path(target_dir)
        self.youtube_url = "https://www.youtube.com/watch?v=86FAWCzIe_4"

    def create_notebook_metadata(self) -> Dict:
        """Create standard notebook metadata for Colab."""
        return {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        }

    def create_markdown_cell(self, content: str) -> Dict:
        """Create a markdown cell."""
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": content.split('\n')
        }

    def create_code_cell(self, code: str) -> Dict:
        """Create a code cell."""
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.split('\n')
        }

    def extract_learning_objectives(self, code: str, filename: str) -> List[str]:
        """Generate learning objectives based on code analysis."""
        objectives = []

        # Analyze code for different patterns
        if '__global__' in code:
            objectives.append("Understand CUDA kernel syntax and execution")
        if 'cudaMalloc' in code or 'cudaMemcpy' in code:
            objectives.append("Learn GPU memory allocation and data transfer")
        if '__shared__' in code:
            objectives.append("Use shared memory for optimization")
        if 'cudaStream' in code:
            objectives.append("Work with CUDA streams for async execution")
        if 'atomicAdd' in code or 'atomic' in code.lower():
            objectives.append("Use atomic operations for thread-safe updates")
        if 'nvtx' in code.lower():
            objectives.append("Profile CUDA code using NVTX")
        if 'cuBLAS' in code or 'cublas' in code.lower():
            objectives.append("Leverage cuBLAS library for optimized operations")
        if 'cuDNN' in code or 'cudnn' in code.lower():
            objectives.append("Use cuDNN for deep learning operations")

        # Add general objective
        if not objectives:
            objectives.append("Understand the CUDA programming concepts demonstrated")

        return objectives

    def generate_title_and_description(self, filename: str, module: str) -> Tuple[str, str]:
        """Generate title and description from filename."""
        # Clean up filename
        name = filename.replace('.cu', '').replace('_', ' ').replace('-', ' ')

        # Create title
        title = name.title()

        # Create description based on keywords
        descriptions = {
            'vector add': 'Learn how to perform parallel vector addition on the GPU.',
            'matmul': 'Implement matrix multiplication using CUDA.',
            'matrix': 'Work with matrix operations on the GPU.',
            'atomic': 'Use atomic operations to handle race conditions.',
            'stream': 'Leverage CUDA streams for concurrent execution.',
            'nvtx': 'Profile your CUDA code using NVIDIA Tools Extension.',
            'tiled': 'Optimize performance using shared memory tiling.',
            'naive': 'Understand the basic implementation approach.',
            'cublas': 'Use the highly optimized cuBLAS library.',
            'cudnn': 'Leverage cuDNN for deep learning operations.',
            'conv': 'Implement convolution operations.',
            'idxing': 'Master CUDA thread indexing and hierarchy.',
        }

        description = "CUDA programming concepts and implementation."
        for key, desc in descriptions.items():
            if key in name.lower():
                description = desc
                break

        return title, description

    def create_header_section(self, title: str, description: str, module: str,
                            filename: str) -> str:
        """Create the notebook header markdown."""
        return f"""# {title}

**FreeCodeCamp CUDA Course - Module {module}**

Original Course: [{self.youtube_url}]({self.youtube_url})
Source File: `{filename}`

---

## Overview

{description}

---
"""

    def create_learning_objectives_section(self, objectives: List[str]) -> str:
        """Create learning objectives section."""
        obj_list = '\n'.join([f"{i+1}. {obj}" for i, obj in enumerate(objectives)])
        return f"""## Learning Objectives

By the end of this notebook, you will:

{obj_list}

---
"""

    def create_setup_section(self) -> List[Dict]:
        """Create setup cells for Colab."""
        cells = []

        # Setup markdown
        cells.append(self.create_markdown_cell(
"""## Setup: Google Colab GPU

First, ensure you have enabled GPU in Colab:
1. Go to **Runtime** → **Change runtime type**
2. Select **T4 GPU** as Hardware accelerator
3. Click **Save**

Let's verify CUDA is available:"""
        ))

        # Check GPU
        cells.append(self.create_code_cell("# Check GPU availability\n!nvidia-smi"))

        # Install nvcc4jupyter
        cells.append(self.create_markdown_cell(
"""Now install the nvcc4jupyter plugin to compile CUDA code inline:"""
        ))

        cells.append(self.create_code_cell(
"""# Install nvcc4jupyter for inline CUDA compilation
!pip install nvcc4jupyter -q
%load_ext nvcc4jupyter"""
        ))

        cells.append(self.create_markdown_cell("---\n"))

        return cells

    def process_cuda_code(self, code: str, filename: str) -> Tuple[str, List[str]]:
        """Process CUDA code and extract key concepts."""
        concepts = []

        # Identify key concepts from code
        if '__global__' in code:
            concepts.append("**Kernel Function**: Uses `__global__` qualifier for GPU execution")
        if '__shared__' in code:
            concepts.append("**Shared Memory**: Fast on-chip memory shared by threads in a block")
        if 'cudaMalloc' in code:
            concepts.append("**Device Memory**: Allocated using `cudaMalloc`")
        if 'cudaMemcpy' in code:
            concepts.append("**Data Transfer**: Uses `cudaMemcpy` between host and device")
        if '<<<' in code and '>>>' in code:
            concepts.append("**Kernel Launch**: Syntax `kernel<<<blocks, threads>>>(...)`")
        if 'cudaDeviceSynchronize' in code:
            concepts.append("**Synchronization**: `cudaDeviceSynchronize()` waits for GPU completion")
        if '__syncthreads' in code:
            concepts.append("**Block Synchronization**: `__syncthreads()` synchronizes threads in a block")

        return code, concepts

    def create_code_section(self, code: str, concepts: List[str], title: str) -> List[Dict]:
        """Create code section with explanation."""
        cells = []

        # Add concepts if any
        if concepts:
            concept_list = '\n'.join([f"- {c}" for c in concepts])
            cells.append(self.create_markdown_cell(
f"""## Key Concepts

{concept_list}

---

## CUDA Implementation"""
            ))
        else:
            cells.append(self.create_markdown_cell("## CUDA Implementation"))

        # Add the CUDA code with %%cu magic
        cuda_code = f"%%cu\n{code}"
        cells.append(self.create_code_cell(cuda_code))

        return cells

    def create_exercise_section(self, filename: str) -> str:
        """Create exercises based on the code type."""
        exercises = {
            'vector_add': """## Exercises

Try these modifications to deepen your understanding:

1. **Change Vector Size**: Modify `N` to different values (100, 1000, 100000000)
   - Observe how execution time changes
   - What happens with very large vectors?

2. **Adjust Block Size**: Try different `BLOCK_SIZE` values (64, 128, 512, 1024)
   - How does it affect performance?
   - What's the maximum allowed?

3. **Add Vector Subtraction**: Create a new kernel for `C = A - B`

4. **Multiple Operations**: Compute `D = (A + B) * (A - B)` in a single kernel""",

            'matmul': """## Exercises

Experiment with these modifications:

1. **Matrix Sizes**: Try different M, N, K dimensions
   - What happens with non-square matrices?
   - Test with very large matrices

2. **Verify Correctness**: Add code to verify the result against CPU computation

3. **Block Size Impact**: Experiment with different BLOCK_SIZE values
   - Measure performance for each

4. **Measure Throughput**: Calculate FLOPS (floating-point operations per second)""",

            'atomic': """## Exercises

Explore atomic operations:

1. **Other Atomic Functions**: Try `atomicMin`, `atomicMax`, `atomicExch`

2. **Performance Comparison**: Measure execution time with and without atomics

3. **Histogram**: Use atomics to build a histogram of random values

4. **Race Condition Demo**: Create a kernel showing race conditions without atomics""",

            'stream': """## Exercises

Practice with CUDA streams:

1. **More Streams**: Create 4+ streams and run operations in parallel

2. **Measure Speedup**: Compare single stream vs multiple streams

3. **Stream Dependencies**: Use events to create complex dependencies

4. **Overlap Computation**: Try overlapping data transfer with computation""",

            'tiled': """## Exercises

Optimize with shared memory:

1. **Different Tile Sizes**: Try 8x8, 32x32, 16x16 tiles
   - How does it affect performance?

2. **Compare with Naive**: Benchmark against naive implementation

3. **Bank Conflicts**: Modify to avoid shared memory bank conflicts

4. **2D Tiling**: Implement 2D block tiling for better data reuse"""
        }

        # Find matching exercise
        for key, exercise in exercises.items():
            if key in filename.lower():
                return exercise

        # Default exercises
        return """## Exercises

Try these modifications:

1. **Modify Parameters**: Change kernel launch parameters and observe effects

2. **Add Error Checking**: Implement CUDA error checking for all API calls

3. **Performance Measurement**: Add timing code to measure execution time

4. **Extend Functionality**: Add new features building on this example"""

    def create_footer_section(self, module: str, notebook_num: int) -> str:
        """Create footer with takeaways and notes."""
        return f"""---

## Key Takeaways

- CUDA enables massive parallelism for compute-intensive tasks
- Proper memory management is crucial for performance
- Understanding the thread hierarchy helps write efficient kernels
- Always synchronize when needed to ensure correctness

---

## Next Steps

Continue to the next notebook in Module {module} to learn more CUDA concepts!

---

## Notes

*Use this space for your learning notes:*


"""

    def convert_cu_file(self, cu_file: Path, module: str, notebook_num: int) -> Dict:
        """Convert a single .cu file to notebook format."""
        print(f"Converting {cu_file.name}...")

        # Read CUDA source code
        with open(cu_file, 'r') as f:
            cuda_code = f.read()

        # Generate notebook components
        filename = cu_file.name
        title, description = self.generate_title_and_description(filename, module)
        objectives = self.extract_learning_objectives(cuda_code, filename)
        processed_code, concepts = self.process_cuda_code(cuda_code, filename)

        # Build notebook cells
        cells = []

        # Header
        cells.append(self.create_markdown_cell(
            self.create_header_section(title, description, module, filename)
        ))

        # Learning objectives
        cells.append(self.create_markdown_cell(
            self.create_learning_objectives_section(objectives)
        ))

        # Setup cells (only for first few notebooks or if needed)
        if notebook_num <= 2:
            cells.extend(self.create_setup_section())
        else:
            cells.append(self.create_markdown_cell(
                "## Setup\n\nMake sure you've completed the setup from the first notebook "
                "(GPU enabled, nvcc4jupyter installed).\n\n---\n"
            ))

        # Code section
        cells.extend(self.create_code_section(processed_code, concepts, title))

        # Exercises
        cells.append(self.create_markdown_cell(
            self.create_exercise_section(filename)
        ))

        # Footer
        cells.append(self.create_markdown_cell(
            self.create_footer_section(module, notebook_num)
        ))

        # Create notebook structure
        notebook = {
            "cells": cells,
            "metadata": self.create_notebook_metadata(),
            "nbformat": 4,
            "nbformat_minor": 0
        }

        return notebook

    def find_cuda_files(self) -> Dict[str, List[Path]]:
        """Find all .cu files organized by module."""
        modules = {
            'module5': [],
            'module6': [],
            'module7': [],
            'module8': [],
            'module9': []
        }

        # Module 5: Writing your First Kernels
        module5_path = self.source_repo / "05_Writing_your_First_Kernels"
        if module5_path.exists():
            modules['module5'] = sorted(module5_path.rglob("*.cu"))

        # Module 6: CUDA APIs
        module6_path = self.source_repo / "06_CUDA_APIs"
        if module6_path.exists():
            modules['module6'] = sorted(module6_path.rglob("*.cu"))

        # Module 7: Faster Matmul
        module7_path = self.source_repo / "07_Faster_Matmul"
        if module7_path.exists():
            modules['module7'] = sorted(module7_path.rglob("*.cu"))

        # Module 8: Triton (will handle .py files separately)
        module8_path = self.source_repo / "08_Triton"
        if module8_path.exists():
            modules['module8'] = sorted(module8_path.rglob("*.cu"))

        # Module 9: PyTorch Extensions
        module9_path = self.source_repo / "09_PyTorch_Extensions"
        if module9_path.exists():
            modules['module9'] = sorted(module9_path.rglob("*.cu"))

        return modules

    def sanitize_filename(self, filepath: Path) -> str:
        """Create a sanitized filename for the notebook."""
        # Get relative path components
        parts = filepath.parts

        # Find the module index
        for i, part in enumerate(parts):
            if 'Writing_your_First_Kernels' in part or 'CUDA_APIs' in part or \
               'Faster_Matmul' in part or 'Triton' in part or 'PyTorch_Extensions' in part:
                relevant_parts = parts[i+1:]
                break
        else:
            relevant_parts = [filepath.stem]

        # Create filename
        name = '_'.join(relevant_parts).replace(' ', '_').replace('.cu', '')
        name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
        return f"{name}.ipynb"

    def create_index_file(self, converted_files: Dict[str, List[str]]) -> None:
        """Create an INDEX.md file with all notebook links."""
        index_content = f"""# FreeCodeCamp CUDA Course - Colab Notebooks

This directory contains Google Colab-compatible notebooks converted from the
[FreeCodeCamp CUDA Course]({self.youtube_url}).

## How to Use

1. Upload notebooks to Google Colab or open them directly from this repository
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run the setup cells in each notebook
4. Follow along with the video course

---

## Module 5: Writing Your First Kernels

"""

        for module, files in sorted(converted_files.items()):
            if not files:
                continue

            module_names = {
                'module5': 'Module 5: Writing Your First Kernels',
                'module6': 'Module 6: CUDA APIs (cuBLAS, cuDNN)',
                'module7': 'Module 7: Optimizing Matrix Multiplication',
                'module8': 'Module 8: Triton',
                'module9': 'Module 9: PyTorch CUDA Extensions'
            }

            if module != 'module5':
                index_content += f"\n## {module_names[module]}\n\n"

            for i, filepath in enumerate(files, 1):
                filename = Path(filepath).name
                title = filename.replace('.ipynb', '').replace('_', ' ').title()
                index_content += f"{i}. [{title}]({module}/{filename})\n"

        index_content += """
---

## Course Resources

- **YouTube Course**: [CUDA Programming on FreeCodeCamp](https://www.youtube.com/watch?v=86FAWCzIe_4)
- **Original Repository**: [Infatoshi/cuda-course](https://github.com/Infatoshi/cuda-course)
- **Your Fork**: Use your forked repository to track changes

---

## Tips for Learning

1. **Run Every Cell**: Don't just read - execute and experiment
2. **Try Exercises**: The exercises reinforce your understanding
3. **Modify Code**: Change parameters and observe the effects
4. **Take Notes**: Use the Notes section at the end of each notebook
5. **Ask Questions**: Use the course Discord or forums if you get stuck

Happy GPU Programming! 🚀
"""

        index_path = self.target_dir / "INDEX.md"
        with open(index_path, 'w') as f:
            f.write(index_content)
        print(f"Created {index_path}")

    def create_readme(self) -> None:
        """Create README.md for the collab-fcc-course directory."""
        readme_content = f"""# FreeCodeCamp CUDA Course - Google Colab Edition

This directory contains Jupyter notebooks adapted for Google Colab from the
[FreeCodeCamp CUDA Programming Course]({self.youtube_url}).

## Quick Start

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
2. **Upload a Notebook**: Upload any `.ipynb` file from this directory
3. **Enable GPU**:
   - Click **Runtime** → **Change runtime type**
   - Select **T4 GPU** as Hardware accelerator
   - Click **Save**
4. **Run the Notebook**: Execute cells sequentially

## Directory Structure

```
collab-fcc-course/
├── module5/     # Writing Your First Kernels
├── module6/     # CUDA APIs (cuBLAS, cuDNN)
├── module7/     # Optimizing Matrix Multiplication
├── module8/     # Triton
├── module9/     # PyTorch CUDA Extensions
├── INDEX.md     # Complete list of all notebooks
└── README.md    # This file
```

## Prerequisites

- Basic Python programming
- Understanding of parallel computing concepts (helpful)
- Google account (for Colab access)

## Course Modules

### Module 5: Writing Your First Kernels
Learn CUDA basics: thread indexing, memory management, kernels, and streams.

### Module 6: CUDA APIs
Leverage optimized libraries like cuBLAS and cuDNN for production performance.

### Module 7: Optimizing Matrix Multiplication
Deep dive into performance optimization techniques and memory hierarchy.

### Module 8: Triton
Explore high-level GPU programming with OpenAI's Triton language.

### Module 9: PyTorch CUDA Extensions
Create custom CUDA operations integrated with PyTorch.

## Learning Path

1. Start with Module 5, Notebook 1 (CUDA Indexing)
2. Work through notebooks sequentially
3. Complete exercises in each notebook
4. Reference the original video course for detailed explanations
5. Experiment with code modifications

## Notes

- **Free GPU Access**: Colab provides free T4 GPU access (with usage limits)
- **Session Limits**: Free tier has 12-hour session limits
- **Save Your Work**: Download modified notebooks to avoid losing changes
- **Runtime Disconnects**: Colab may disconnect after inactivity

## Attribution

- **Original Course**: FreeCodeCamp CUDA Programming Course
- **Course Author**: Check the original repository for author information
- **Notebooks**: Converted for educational use with Google Colab

## Additional Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Course Discord/Community]: Check the original course for community links

---

**Happy GPU Programming!** 🚀
"""

        readme_path = self.target_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"Created {readme_path}")

    def convert_all(self) -> None:
        """Convert all CUDA files to Colab notebooks."""
        print("="*60)
        print("CUDA to Colab Notebook Converter")
        print("="*60)
        print(f"Source: {self.source_repo}")
        print(f"Target: {self.target_dir}")
        print()

        # Find all CUDA files
        modules = self.find_cuda_files()

        # Track converted files for index
        converted_files = {}

        # Convert each module
        for module_name, cu_files in modules.items():
            if not cu_files:
                print(f"No .cu files found for {module_name}")
                continue

            print(f"\n{'='*60}")
            print(f"Processing {module_name.upper()} ({len(cu_files)} files)")
            print(f"{'='*60}")

            module_dir = self.target_dir / module_name
            module_dir.mkdir(parents=True, exist_ok=True)

            converted_files[module_name] = []

            for i, cu_file in enumerate(cu_files, 1):
                try:
                    # Convert to notebook
                    notebook = self.convert_cu_file(cu_file, module_name[-1], i)

                    # Generate output filename
                    output_name = self.sanitize_filename(cu_file)
                    output_path = module_dir / output_name

                    # Write notebook
                    with open(output_path, 'w') as f:
                        json.dump(notebook, f, indent=2)

                    print(f"  ✓ Created: {output_path.relative_to(self.target_dir)}")
                    converted_files[module_name].append(str(output_path.relative_to(self.target_dir)))

                except Exception as e:
                    print(f"  ✗ Error converting {cu_file.name}: {e}")

        # Create index and README
        print(f"\n{'='*60}")
        print("Creating documentation files...")
        print(f"{'='*60}")
        self.create_index_file(converted_files)
        self.create_readme()

        # Summary
        total_converted = sum(len(files) for files in converted_files.values())
        print(f"\n{'='*60}")
        print(f"Conversion Complete!")
        print(f"{'='*60}")
        print(f"Total notebooks created: {total_converted}")
        print(f"\nStart learning with: {self.target_dir}/module5/")
        print(f"See INDEX.md for full notebook list")


def main():
    """Main entry point."""
    # Define paths
    cuda_course_repo = "/Users/negoel/code/mywork/github/neerajgoel82/cuda-course-fcc"
    output_dir = "collab-fcc-course"

    # Create converter and run
    converter = CUDAToColabConverter(cuda_course_repo, output_dir)
    converter.convert_all()


if __name__ == "__main__":
    main()

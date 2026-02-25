#!/usr/bin/env python3
"""
Verify that notebooks were updated correctly by checking code content.
"""

import json
from pathlib import Path

COLAB_NOTEBOOKS = Path(__file__).parent / "colab" / "notebooks"

# Sample notebooks to check from each phase
SAMPLE_NOTEBOOKS = [
    "phase1/00-setup-verification.ipynb",
    "phase2/10_tiled_matrix_multiplication.ipynb",
    "phase3/17_histogram.ipynb",
    "phase5/24_gemm_optimized.ipynb",
    "phase8/45_raytracer.ipynb",
    "phase9/55_wmma_gemm.ipynb",
]

def check_notebook(notebook_path):
    """Check if a notebook has real CUDA code."""
    with open(notebook_path, 'r') as f:
        notebook_data = json.load(f)

    # Find first %%cu cell
    for cell in notebook_data['cells']:
        if cell.get('cell_type') == 'code':
            source = cell.get('source', '')
            if isinstance(source, list):
                source = ''.join(source)

            if source.strip().startswith('%%cu'):
                # Check for real implementation markers
                has_kernel = '__global__' in source
                has_error_check = 'CUDA_CHECK' in source
                has_malloc = 'cudaMalloc' in source or 'malloc' in source
                has_real_logic = len(source) > 500  # Real code is substantial

                # Check it's not the generic template
                is_generic = 'data[idx] = data[idx] * 2.0f' in source and len(source) < 2000

                return {
                    'has_kernel': has_kernel,
                    'has_error_check': has_error_check,
                    'has_malloc': has_malloc,
                    'has_real_logic': has_real_logic,
                    'is_not_generic': not is_generic,
                    'code_size': len(source),
                    'lines': source.count('\n') + 1
                }

    return None

def main():
    print("=" * 70)
    print("Notebook Update Verification")
    print("=" * 70)
    print()

    print("Checking sample notebooks for real implementations...\n")

    all_good = True
    for notebook_rel in SAMPLE_NOTEBOOKS:
        notebook_path = COLAB_NOTEBOOKS / notebook_rel

        print(f"Checking: {notebook_rel}")

        if not notebook_path.exists():
            print(f"  ERROR: Notebook not found")
            all_good = False
            continue

        result = check_notebook(notebook_path)

        if result is None:
            print(f"  ERROR: No %%cu cell found")
            all_good = False
            continue

        print(f"  Code size: {result['code_size']} bytes ({result['lines']} lines)")
        print(f"  Has kernel: {'✓' if result['has_kernel'] else '✗'}")
        print(f"  Has error checking: {'✓' if result['has_error_check'] else '✗'}")
        print(f"  Has memory management: {'✓' if result['has_malloc'] else '✗'}")
        print(f"  Has real implementation: {'✓' if result['has_real_logic'] else '✗'}")
        print(f"  Not generic template: {'✓' if result['is_not_generic'] else '✗'}")

        if all([result['has_kernel'], result['has_error_check'],
                result['has_malloc'], result['has_real_logic'],
                result['is_not_generic']]):
            print(f"  Status: PASS ✓")
        else:
            print(f"  Status: FAIL ✗")
            all_good = False

        print()

    print("=" * 70)
    if all_good:
        print("Verification: ALL CHECKS PASSED ✓")
    else:
        print("Verification: SOME CHECKS FAILED ✗")
    print("=" * 70)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Smart verification that filters out false positives.
"""

import glob

# Files that legitimately don't need custom kernels (they use libraries)
LIBRARY_WRAPPERS = [
    '25_cublas_integration.cu',  # Uses cuBLAS
    '29_thrust_examples.cu',     # Uses Thrust
    '41_cufft_demo.cu',          # Uses cuFFT
    '42_cusparse_demo.cu',       # Uses cuSPARSE
    '43_curand_demo.cu',         # Uses cuRAND
    '34_p2p_transfer.cu',        # Uses cudaMemcpy peer-to-peer
    '35_nccl_collectives.cu',    # Uses NCCL
    '02_device_query.cu',        # Just queries device
    '06_memory_basics_and_data_transfer.cu',  # Just transfers
    '01_memory_bandwidth.cu',    # Just transfers
]

def check_cu_file(filepath):
    """Check if .cu file has real issues."""
    filename = filepath.split('/')[-1]

    # Skip library wrappers
    if filename in LIBRARY_WRAPPERS:
        return None

    with open(filepath, 'r') as f:
        content = f.read()

    # Check for generic template
    if 'data[idx] = data[idx] * 2.0f' in content:
        return "GENERIC TEMPLATE"

    return None

def check_notebook(filepath):
    """Check if notebook has real issues."""
    import json

    with open(filepath, 'r') as f:
        try:
            notebook = json.load(f)
        except:
            return "JSON ERROR"

    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if '%%cu' in source and 'data[idx] = data[idx] * 2.0f' in source:
                return "GENERIC TEMPLATE"

    return None

def main():
    print("="*70)
    print("REAL ISSUES CHECK (filtering out false positives)")
    print("="*70)
    print()

    # Check .cu files
    print("LOCAL .CU FILES:")
    print("-"*70)

    cu_files = sorted(glob.glob('local/phase*/*.cu'))
    real_issues_cu = {}

    for filepath in cu_files:
        issue = check_cu_file(filepath)
        if issue:
            real_issues_cu[filepath] = issue

    if real_issues_cu:
        print(f"❌ {len(real_issues_cu)} files with REAL issues:\n")
        for filepath, issue in real_issues_cu.items():
            print(f"  {filepath}")
            print(f"    → {issue}\n")
    else:
        print(f"✅ All {len(cu_files)} files are OK!\n")

    # Check notebooks
    print("-"*70)
    print("COLAB NOTEBOOKS:")
    print("-"*70)

    notebooks = sorted(glob.glob('colab/notebooks/phase*/*.ipynb'))
    real_issues_nb = {}

    for filepath in notebooks:
        issue = check_notebook(filepath)
        if issue:
            real_issues_nb[filepath] = issue

    if real_issues_nb:
        print(f"❌ {len(real_issues_nb)} notebooks with REAL issues:\n")
        for filepath, issue in real_issues_nb.items():
            print(f"  {filepath}")
            print(f"    → {issue}\n")
    else:
        print(f"✅ All {len(notebooks)} notebooks are OK!\n")

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f".cu files: {len(cu_files)} total, {len(real_issues_cu)} with real issues")
    print(f"Notebooks: {len(notebooks)} total, {len(real_issues_nb)} with real issues")
    print()

    if real_issues_cu or real_issues_nb:
        print(f"❌ {len(real_issues_cu) + len(real_issues_nb)} files need fixing")
        return 1
    else:
        print("✅ ALL FILES VERIFIED!")
        return 0

if __name__ == "__main__":
    exit(main())

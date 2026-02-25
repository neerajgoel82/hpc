#!/usr/bin/env python3
"""
Comprehensive verification of all CUDA files and notebooks.
Checks for empty files, generic templates, and placeholder content.
"""

import os
import json
import glob

def check_cu_file(filepath):
    """Check a .cu file for issues."""
    issues = []

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        size = len(content)

        # Check if empty
        if size == 0:
            issues.append("EMPTY FILE (0 bytes)")
            return issues

        # Check if too small (likely incomplete)
        if size < 500:
            issues.append(f"TOO SMALL ({size} bytes)")

        # Check for generic placeholder patterns
        if 'data[idx] = data[idx] * 2.0f' in content:
            issues.append("GENERIC TEMPLATE (data[idx] * 2)")

        if 'printf("Example kernel' in content:
            issues.append("PLACEHOLDER TEXT")

        # Check for TODO/FIXME
        if 'TODO' in content or 'FIXME' in content:
            issues.append("HAS TODO/FIXME")

        # Check if has main function
        if 'int main(' not in content:
            issues.append("NO MAIN FUNCTION")

        # Check if has at least one kernel
        if '__global__' not in content:
            issues.append("NO CUDA KERNEL")

    except Exception as e:
        issues.append(f"ERROR READING: {e}")

    return issues

def check_notebook(filepath):
    """Check a notebook for issues."""
    issues = []

    try:
        with open(filepath, 'r') as f:
            notebook = json.load(f)

        # Count code cells with %%cu
        cuda_cells = 0
        has_generic = False

        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                if '%%cu' in source:
                    cuda_cells += 1

                    # Check for generic template
                    if 'data[idx] = data[idx] * 2.0f' in source:
                        has_generic = True

                    if 'printf("Example kernel' in source:
                        has_generic = True

        if cuda_cells == 0:
            issues.append("NO CUDA CODE CELLS")

        if has_generic:
            issues.append("HAS GENERIC TEMPLATE")

    except json.JSONDecodeError as e:
        issues.append(f"JSON ERROR: {e}")
    except Exception as e:
        issues.append(f"ERROR: {e}")

    return issues

def main():
    print("="*70)
    print("COMPREHENSIVE FILE VERIFICATION")
    print("="*70)
    print()

    # Check .cu files
    print("Checking .cu files in local/...")
    print("-" * 70)

    cu_files = sorted(glob.glob('local/phase*/*.cu'))
    cu_issues = {}

    for filepath in cu_files:
        issues = check_cu_file(filepath)
        if issues:
            cu_issues[filepath] = issues

    if cu_issues:
        print(f"❌ Found issues in {len(cu_issues)} .cu files:\n")
        for filepath, issues in cu_issues.items():
            print(f"  {filepath}")
            for issue in issues:
                print(f"    - {issue}")
            print()
    else:
        print(f"✅ All {len(cu_files)} .cu files are OK!\n")

    # Check notebooks
    print("-" * 70)
    print("Checking notebooks in colab/notebooks/...")
    print("-" * 70)

    notebooks = sorted(glob.glob('colab/notebooks/phase*/*.ipynb'))
    notebook_issues = {}

    for filepath in notebooks:
        issues = check_notebook(filepath)
        if issues:
            notebook_issues[filepath] = issues

    if notebook_issues:
        print(f"❌ Found issues in {len(notebook_issues)} notebooks:\n")
        for filepath, issues in notebook_issues.items():
            print(f"  {filepath}")
            for issue in issues:
                print(f"    - {issue}")
            print()
    else:
        print(f"✅ All {len(notebooks)} notebooks are OK!\n")

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total .cu files checked: {len(cu_files)}")
    print(f"  Issues found: {len(cu_issues)}")
    print(f"  Clean: {len(cu_files) - len(cu_issues)}")
    print()
    print(f"Total notebooks checked: {len(notebooks)}")
    print(f"  Issues found: {len(notebook_issues)}")
    print(f"  Clean: {len(notebooks) - len(notebook_issues)}")
    print()

    if cu_issues or notebook_issues:
        print("❌ VERIFICATION FAILED - Issues found")
        return 1
    else:
        print("✅ VERIFICATION PASSED - All files OK!")
        return 0

if __name__ == "__main__":
    exit(main())

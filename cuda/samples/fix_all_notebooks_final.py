#!/usr/bin/env python3
"""
Fix all CUDA notebooks by replacing template code with real implementations
from local .cu files.
"""

import json
import os
import re
from pathlib import Path

# Base paths
SCRIPT_DIR = Path(__file__).parent
COLAB_NOTEBOOKS = SCRIPT_DIR / "colab" / "notebooks"
LOCAL_SAMPLES = SCRIPT_DIR / "local"

# Mapping of notebooks to fix with their corresponding .cu files
NOTEBOOKS_TO_FIX = {
    # Phase 1
    "phase1/00-setup-verification.ipynb": "phase1/02_device_query.cu",

    # Phase 2
    "phase2/10_tiled_matrix_multiplication.ipynb": "phase3/10_tiled_matrix_multiplication.cu",

    # Phase 3
    "phase3/14_occupancy_tuning.ipynb": "phase3/14_occupancy_tuning.cu",
    "phase3/16_prefix_sum.ipynb": "phase3/16_prefix_sum.cu",
    "phase3/17_histogram.ipynb": "phase3/17_histogram.cu",

    # Phase 4
    "phase4/18_texture_memory.ipynb": "phase4/18_texture_memory.cu",
    "phase4/19_constant_memory.ipynb": "phase4/19_constant_memory.cu",
    "phase4/20_zero_copy.ipynb": "phase4/20_zero_copy.cu",
    "phase4/21_atomics.ipynb": "phase4/21_atomics.cu",
    "phase4/22_cooperative_groups.ipynb": "phase4/22_cooperative_groups.cu",

    # Phase 5
    "phase5/24_gemm_optimized.ipynb": "phase5/24_gemm_optimized.cu",
    "phase5/25_cublas_integration.ipynb": "phase5/25_cublas_integration.cu",
    "phase5/26_matrix_transpose.ipynb": "phase5/26_matrix_transpose.cu",
    "phase5/27_bitonic_sort.ipynb": "phase5/27_bitonic_sort.cu",
    "phase5/28_radix_sort.ipynb": "phase5/28_radix_sort.cu",
    "phase5/29_thrust_examples.ipynb": "phase5/29_thrust_examples.cu",

    # Phase 6
    "phase6/30_streams_basic.ipynb": "phase6/30_streams_basic.cu",
    "phase6/31_async_pipeline.ipynb": "phase6/31_async_pipeline.cu",
    "phase6/32_events_timing.ipynb": "phase6/32_events_timing.cu",
    "phase6/33_multi_gpu_basic.ipynb": "phase6/33_multi_gpu_basic.cu",
    "phase6/34_p2p_transfer.ipynb": "phase6/34_p2p_transfer.cu",
    "phase6/35_nccl_collectives.ipynb": "phase6/35_nccl_collectives.cu",

    # Phase 7
    "phase7/36_profiling_demo.ipynb": "phase7/36_profiling_demo.cu",
    "phase7/37_debugging_cuda.ipynb": "phase7/37_debugging_cuda.cu",
    "phase7/38_kernel_fusion.ipynb": "phase7/38_kernel_fusion.cu",
    "phase7/39_fast_math.ipynb": "phase7/39_fast_math.cu",

    # Phase 8
    "phase8/41_cufft_demo.ipynb": "phase8/41_cufft_demo.cu",
    "phase8/42_cusparse_demo.ipynb": "phase8/42_cusparse_demo.cu",
    "phase8/43_curand_demo.ipynb": "phase8/43_curand_demo.cu",
    "phase8/44_image_processing.ipynb": "phase8/44_image_processing.cu",
    "phase8/45_raytracer.ipynb": "phase8/45_raytracer.cu",
    "phase8/46_nbody_simulation.ipynb": "phase8/46_nbody_simulation.cu",
    "phase8/47_neural_network.ipynb": "phase8/47_neural_network.cu",
    "phase8/48_molecular_dynamics.ipynb": "phase8/48_molecular_dynamics.cu",
    "phase8/49_option_pricing.ipynb": "phase8/49_option_pricing.cu",

    # Phase 9
    "phase9/50_dynamic_parallelism.ipynb": "phase9/50_dynamic_parallelism.cu",
    "phase9/51_cuda_graphs.ipynb": "phase9/51_cuda_graphs.cu",
    "phase9/52_mps_demo.ipynb": "phase9/52_mps_demo.cu",
    "phase9/53_mixed_precision.ipynb": "phase9/53_mixed_precision.cu",
    "phase9/54_tensor_cores.ipynb": "phase9/54_tensor_cores.cu",
    "phase9/55_wmma_gemm.ipynb": "phase9/55_wmma_gemm.cu",
}


def read_cu_file(cu_path):
    """Read and return the content of a .cu file."""
    with open(cu_path, 'r') as f:
        return f.read()


def find_first_cu_cell(notebook_data):
    """Find the index of the first cell that starts with %%cu."""
    for idx, cell in enumerate(notebook_data['cells']):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if source:
                # Handle both list and string source
                first_line = source[0] if isinstance(source, list) else source.split('\n')[0]
                if first_line.strip().startswith('%%cu'):
                    return idx
    return None


def update_notebook(notebook_path, cu_code):
    """Update the first %%cu cell in a notebook with new code."""
    with open(notebook_path, 'r') as f:
        notebook_data = json.load(f)

    cell_idx = find_first_cu_cell(notebook_data)

    if cell_idx is None:
        print(f"  WARNING: No %%cu cell found in {notebook_path}")
        return False

    # Prepare the new source with %%cu magic at the top
    new_source = f"%%cu\n{cu_code}"

    # Update the cell
    notebook_data['cells'][cell_idx]['source'] = new_source

    # Write back the notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook_data, f, indent=1)

    return True


def main():
    """Main function to fix all notebooks."""
    print("=" * 70)
    print("CUDA Notebook Fixer - Updating 41 notebooks with real implementations")
    print("=" * 70)
    print()

    success_count = 0
    failed_count = 0
    skipped_count = 0

    for notebook_rel, cu_rel in NOTEBOOKS_TO_FIX.items():
        notebook_path = COLAB_NOTEBOOKS / notebook_rel
        cu_path = LOCAL_SAMPLES / cu_rel

        print(f"Processing: {notebook_rel}")
        print(f"  Source: {cu_rel}")

        # Check if files exist
        if not notebook_path.exists():
            print(f"  ERROR: Notebook not found: {notebook_path}")
            failed_count += 1
            print()
            continue

        if not cu_path.exists():
            print(f"  ERROR: .cu file not found: {cu_path}")
            failed_count += 1
            print()
            continue

        # Read the .cu file
        try:
            cu_code = read_cu_file(cu_path)
            print(f"  Read {len(cu_code)} bytes from .cu file")
        except Exception as e:
            print(f"  ERROR reading .cu file: {e}")
            failed_count += 1
            print()
            continue

        # Update the notebook
        try:
            if update_notebook(notebook_path, cu_code):
                print(f"  SUCCESS: Updated notebook")
                success_count += 1
            else:
                print(f"  SKIPPED: No %%cu cell to update")
                skipped_count += 1
        except Exception as e:
            print(f"  ERROR updating notebook: {e}")
            failed_count += 1

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successfully updated: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Total processed: {len(NOTEBOOKS_TO_FIX)}")

    if success_count == len(NOTEBOOKS_TO_FIX):
        print("\n✓ All notebooks successfully updated!")
    elif success_count > 0:
        print(f"\n⚠ {success_count}/{len(NOTEBOOKS_TO_FIX)} notebooks updated")
    else:
        print("\n✗ No notebooks were updated")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    exit(main())

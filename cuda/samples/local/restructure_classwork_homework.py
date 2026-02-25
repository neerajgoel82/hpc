#!/usr/bin/env python3
"""
Restructure cuda/samples/local phases into classwork/ and homework/ layout.
- Moves all *.cu from each phase root into phase/classwork/
- Creates homework/*.cu stubs for each classwork file (minimal compiling stub).
Run from repo root or from cuda/samples/local.
"""

import os
import shutil

LOCAL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__))
)
PHASES = [f"phase{i}" for i in range(1, 10)]

HOMEWORK_STUB = '''// Homework: {basename}
// Complete based on classwork/{basename}
//
// Instructions:
// 1. Read the corresponding classwork file first
// 2. Implement the exercises below
// 3. Build: make homework (or make both)
// 4. Run your executable from the phase directory

#include <stdio.h>
#include <cuda_runtime.h>

int main() {{
    printf("Homework: {name} - Implement the exercises from classwork\\n");
    return 0;
}}
'''


def main():
    for phase in PHASES:
        phase_path = os.path.join(LOCAL_DIR, phase)
        if not os.path.isdir(phase_path):
            continue
        classwork_dir = os.path.join(phase_path, "classwork")
        homework_dir = os.path.join(phase_path, "homework")
        os.makedirs(classwork_dir, exist_ok=True)
        os.makedirs(homework_dir, exist_ok=True)

        # Move *.cu from phase root into classwork/ (skip if already in classwork)
        for name in os.listdir(phase_path):
            if not name.endswith(".cu"):
                continue
            src = os.path.join(phase_path, name)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(classwork_dir, name)
            if os.path.exists(dst):
                continue
            shutil.move(src, dst)
            print(f"  Moved {phase}/{name} -> {phase}/classwork/")

        # Create homework stub for each classwork .cu
        for name in sorted(os.listdir(classwork_dir)):
            if not name.endswith(".cu"):
                continue
            cw_path = os.path.join(classwork_dir, name)
            hw_path = os.path.join(homework_dir, name)
            if os.path.exists(hw_path):
                continue
            name_no_ext = os.path.splitext(name)[0]
            with open(hw_path, "w") as f:
                f.write(HOMEWORK_STUB.format(basename=name, name=name_no_ext))
            print(f"  Created {phase}/homework/{name}")

    print("Done. Update each phase Makefile to build classwork/ and homework/.")


if __name__ == "__main__":
    main()

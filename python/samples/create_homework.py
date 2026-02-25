#!/usr/bin/env python3
"""
Script to extract exercises from classwork files and create homework files for Python
"""

import os
import re
import sys

SAMPLES_DIR = "python/samples"
PHASES = [
    "phase1-foundations",
    "phase2-intermediate",
    "phase3-oop",
    "phase4-advanced",
    "phase5-datascience",
    "phase6-pytorch"
]

def extract_exercises(content):
    """Extract TODO/exercise sections from file content"""
    # Look for sections with TODO, Quick Challenge, Exercise, Practice, etc.
    patterns = [
        r'# TODO:.*',
        r'# Exercise:.*',
        r'# Practice:.*',
        r'# Quick Challenge:.*(?:\n#.*)*',
        r'# Your turn:.*(?:\n#.*)*',
        r'# Try this:.*(?:\n#.*)*',
    ]

    exercises = []
    for pattern in patterns:
        matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
        exercises.extend(matches)

    if exercises:
        return "\n".join(exercises)
    return None

def create_homework_file(classwork_file, homework_file, exercises, filename):
    """Create homework file with template and exercises. Always writes a file (stub if no exercises)."""

    # Extract base name without extension
    base_name = os.path.splitext(filename)[0]
    exercise_block = exercises if exercises else "# TODO: Implement based on the corresponding classwork file"

    template = f'''"""
Homework: {filename}

Complete the exercises below based on the concepts from {filename}
in the classwork folder.

Instructions:
1. Read the corresponding classwork file first
2. Implement the solutions below
3. Run: python3 {filename}
4. Test your solutions
"""

{exercise_block}

# Your code here

if __name__ == "__main__":
    print(f"Homework: {base_name}")
    print("Implement the exercises above")

    # TODO: Implement your solutions here
    pass
'''

    os.makedirs(os.path.dirname(homework_file) or '.', exist_ok=True)
    with open(homework_file, 'w') as f:
        f.write(template)

def process_phase(phase, recursive=False):
    """Process all files in a phase. If recursive=True, walk classwork subdirs (phase5/phase6)."""
    phase_dir = os.path.join(SAMPLES_DIR, phase)
    classwork_dir = os.path.join(phase_dir, "classwork")
    homework_dir = os.path.join(phase_dir, "homework")

    if not os.path.exists(classwork_dir):
        print(f"  Skipping {phase} (classwork not found)")
        return

    if recursive:
        # Walk classwork recursively: preserve path under homework
        for root, _dirs, files in os.walk(classwork_dir):
            rel_root = os.path.relpath(root, classwork_dir)
            for filename in sorted(files):
                if not filename.endswith('.py'):
                    continue
                classwork_file = os.path.join(root, filename)
                if not os.path.isfile(classwork_file):
                    continue
                # homework path mirrors classwork path
                if rel_root == '.':
                    homework_file = os.path.join(homework_dir, filename)
                else:
                    subdir = os.path.join(homework_dir, rel_root)
                    os.makedirs(subdir, exist_ok=True)
                    homework_file = os.path.join(subdir, filename)
                _process_one_file(classwork_file, homework_file, filename, phase)
        return

    # Flat structure: only .py files directly in classwork/
    for filename in sorted(os.listdir(classwork_dir)):
        if not filename.endswith('.py'):
            continue
        classwork_file = os.path.join(classwork_dir, filename)
        if not os.path.isfile(classwork_file):
            continue
        homework_file = os.path.join(homework_dir, filename)
        _process_one_file(classwork_file, homework_file, filename, phase)


def _process_one_file(classwork_file, homework_file, filename, phase):
    """Read classwork, extract exercises, write homework (always create stub)."""
    try:
        with open(classwork_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"  Error reading {filename}: {e}")
        return
    exercises = extract_exercises(content)
    # Always create homework file (stub if no exercises)
    create_homework_file(classwork_file, homework_file, exercises, filename)
    rel_path = os.path.relpath(homework_file, os.path.join(SAMPLES_DIR, phase))
    if not rel_path.startswith("homework"):
        rel_path = os.path.join("homework", rel_path)
    print(f"  Created {rel_path}")

def main():
    print("Creating homework files from exercises...\n")

    # Phases with topic subdirs (classwork/01-topic/, etc.) need recursive processing
    recursive_phases = ["phase5-datascience", "phase6-pytorch"]

    for phase in PHASES:
        print(f"Processing {phase}...")
        process_phase(phase, recursive=(phase in recursive_phases))

    print("\nHomework files created successfully!")

if __name__ == "__main__":
    main()

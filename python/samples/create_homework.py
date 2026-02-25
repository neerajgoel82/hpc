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
    """Create homework file with template and exercises"""

    # Extract base name without extension
    base_name = os.path.splitext(filename)[0]

    # Create homework file template
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

{exercises if exercises else "# No exercises found"}

# Your code here

if __name__ == "__main__":
    print(f"Homework: {base_name}")
    print("Implement the exercises above")

    # TODO: Implement your solutions here
    pass
'''

    with open(homework_file, 'w') as f:
        f.write(template)

def process_phase(phase):
    """Process all files in a phase"""
    phase_dir = os.path.join(SAMPLES_DIR, phase)
    classwork_dir = os.path.join(phase_dir, "classwork")
    homework_dir = os.path.join(phase_dir, "homework")

    if not os.path.exists(classwork_dir):
        print(f"  Skipping {phase} (classwork not found)")
        return

    # Process each .py file in classwork (not in subdirectories)
    for filename in sorted(os.listdir(classwork_dir)):
        if not filename.endswith('.py'):
            continue

        classwork_file = os.path.join(classwork_dir, filename)
        homework_file = os.path.join(homework_dir, filename)

        # Skip if it's inside a subdirectory
        if not os.path.isfile(classwork_file):
            continue

        # Read classwork file
        try:
            with open(classwork_file, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue

        # Extract exercises
        exercises = extract_exercises(content)

        if exercises:
            # Create homework file
            create_homework_file(classwork_file, homework_file, exercises, filename)
            print(f"  Created homework/{filename}")
        else:
            print(f"  No exercises found in {filename}")

def main():
    print("Creating homework files from exercises...\n")

    for phase in PHASES:
        print(f"Processing {phase}...")
        process_phase(phase)

    print("\nHomework files created successfully!")

if __name__ == "__main__":
    main()

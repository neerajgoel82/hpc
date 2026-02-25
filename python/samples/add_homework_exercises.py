#!/usr/bin/env python3
"""
Add meaningful exercises to homework files that are currently templates.
Reads the corresponding classwork file, extracts topics/sections, and generates
concrete exercises with starter code.
Run from repo root: python3 python/samples/add_homework_exercises.py
"""

import os
import re

SAMPLES = os.path.join(os.path.dirname(__file__))


def get_topics_from_classwork(content):
    """Extract topic list from docstring (Topics: or - bullet list)."""
    topics = []
    # Topics: line
    m = re.search(r"Topics?:\s*\n((?:\s*[-*]\s*.+\n?)+)", content, re.IGNORECASE)
    if m:
        for line in m.group(1).strip().split("\n"):
            line = re.sub(r"^\s*[-*]\s*", "", line).strip()
            if line:
                topics.append(line)
    if not topics:
        # Try "Learning Objectives:" or first ## section
        m = re.search(r"(?:Learning Objectives?|Topics?):\s*\n((?:\s*[-*]\s*.+\n?)+)", content, re.IGNORECASE)
        if m:
            for line in m.group(1).strip().split("\n"):
                line = re.sub(r"^\s*[-*]\s*", "", line).strip()
                if line:
                    topics.append(line)
    return topics[:6]  # at most 6


def get_title_from_classwork(content):
    """First line of docstring or filename."""
    m = re.search(r'"""\s*\n?\s*([^\n"]+)', content)
    if m:
        return m.group(1).strip()
    return ""


def generate_exercises_from_topics(topics, title, filename):
    """Generate 3-4 exercise descriptions and function names from topics."""
    if not topics:
        return [
            ("Implement the main concept from the classwork file.", "exercise_1"),
            ("Add a small variation or extension of the classwork example.", "exercise_2"),
        ]
    exercises = []
    for i, t in enumerate(topics[:4], 1):
        # Turn topic into an exercise
        t_clean = t.strip("-* ").strip()
        if not t_clean:
            continue
        # e.g. "Tensor creation methods" -> "Create tensors using at least 3 different methods (e.g. tensor(), zeros(), arange())."
        action = "Implement" if not re.match(r"^(create|write|use|explore|practice)", t_clean, re.I) else ""
        exercises.append((f"{action} {t_clean}.", f"exercise_{i}"))
    if len(exercises) < 2:
        exercises.append(("Test your implementation with a small example.", "exercise_2"))
    return exercises


def write_homework_with_exercises(classwork_path, homework_path, content_from_classwork):
    """Write homework file with exercises and starter code."""
    topics = get_topics_from_classwork(content_from_classwork)
    title = get_title_from_classwork(content_from_classwork)
    filename = os.path.basename(homework_path)
    base = os.path.splitext(filename)[0]
    exercises = generate_exercises_from_topics(topics, title, filename)

    lines = [
        '"""',
        f"Homework: {filename}",
        "",
        "Complete the exercises below based on the concepts from the classwork file.",
        "",
        "Instructions:",
        "1. Read the corresponding classwork file first.",
        "2. Implement each exercise_* function below.",
        f"3. Run: python3 {filename}",
"4. The main block calls each exercise; implement the function bodies to see output.",
'"""',
        "",
    ]
    for i, (desc, fn) in enumerate(exercises, 1):
        lines.append(f"# Exercise {i}: {desc}")
        lines.append(f"def {fn}():")
        lines.append(f'    """{desc}"""')
        lines.append("    pass  # Your code here")
        lines.append("")
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append(f'    print("Homework: {base}")')
    for _, fn in exercises:
        lines.append(f"    {fn}()")
    lines.append("")

    with open(homework_path, "w") as f:
        f.write("\n".join(lines))


def process_phase(phase_dir, recursive=False):
    """Process all template homework files in a phase."""
    classwork_dir = os.path.join(phase_dir, "classwork")
    homework_dir = os.path.join(phase_dir, "homework")
    if not os.path.isdir(classwork_dir) or not os.path.isdir(homework_dir):
        return 0
    count = 0
    if recursive:
        for root, _, files in os.walk(homework_dir):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                hw_path = os.path.join(root, fn)
                rel = os.path.relpath(hw_path, homework_dir)
                cw_path = os.path.join(classwork_dir, rel)
                if not os.path.isfile(cw_path):
                    continue
                with open(hw_path) as f:
                    if "Implement based on the corresponding classwork" not in f.read():
                        continue
                with open(cw_path) as f:
                    cw_content = f.read()
                write_homework_with_exercises(cw_path, hw_path, cw_content)
                count += 1
                print(f"  Updated {os.path.relpath(hw_path, phase_dir)}")
    else:
        for fn in sorted(os.listdir(homework_dir)):
            if not fn.endswith(".py"):
                continue
            hw_path = os.path.join(homework_dir, fn)
            cw_path = os.path.join(classwork_dir, fn)
            if not os.path.isfile(cw_path):
                continue
            with open(hw_path) as f:
                if "Implement based on the corresponding classwork" not in f.read():
                    continue
            with open(cw_path) as f:
                cw_content = f.read()
            write_homework_with_exercises(cw_path, hw_path, cw_content)
            count += 1
            print(f"  Updated homework/{fn}")
    return count


def main():
    phases = [
        "phase2-intermediate",
        "phase3-oop",
        "phase4-advanced",
        "phase5-datascience",
        "phase6-pytorch",
    ]
    total = 0
    for phase in phases:
        path = os.path.join(SAMPLES, phase)
        if not os.path.isdir(path):
            continue
        print(f"Processing {phase}...")
        recursive = phase in ("phase5-datascience", "phase6-pytorch")
        n = process_phase(path, recursive=recursive)
        total += n
    print(f"\nUpdated {total} homework files with meaningful exercises.")


if __name__ == "__main__":
    main()

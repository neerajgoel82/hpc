#!/usr/bin/env python3
"""
Create missing classwork + homework stub files for phase2, phase3, phase4
so every topic listed in the phase README has a classwork and homework file.
"""

import os

BASE = os.path.abspath(os.path.dirname(__file__))

# (phase_dir, [list of .py basenames])
STUBS = [
    ("phase2-intermediate", [
        "17_function_parameters.py", "18_return_values.py", "19_scope.py",
        "20_lambda_functions.py", "21_recursion.py", "project_text_analyzer.py",
        "23_file_reading.py", "24_file_writing.py", "25_csv_files.py",
        "project_file_organizer.py",
        "27_importing.py", "28_standard_library.py", "29_creating_modules.py",
        "30_packages.py", "31_virtual_environments.py", "project_cli_tool.py",
    ]),
    ("phase3-oop", [
        "33_attributes_methods.py", "34_constructors.py", "35_class_variables.py",
        "36_encapsulation.py", "project_bank_account.py",
        "38_polymorphism.py", "39_magic_methods.py", "40_properties.py",
        "41_composition.py", "project_game_characters.py",
    ]),
    ("phase4-advanced", [
        "42_iterators.py", "44_decorators_basics.py", "45_decorators_advanced.py",
        "46_args_kwargs.py", "project_data_pipeline.py",
        "47_regex.py", "48_datetime.py", "49_collections.py",
        "50_comprehensions_advanced.py", "51_file_operations.py",
        "project_log_analyzer.py",
        "52_debugging.py", "53_unittest.py", "54_pytest.py", "55_logging.py",
        "56_best_practices.py", "project_tested_library.py",
    ]),
]

CLASSWORK_TEMPLATE = '''"""
{title} - See phase README for description and learning objectives.
"""

if __name__ == "__main__":
    print("See phase README. Implement exercises in the homework file.")
'''

HOMEWORK_TEMPLATE = '''"""
Homework: {filename}

Complete the exercises below based on the concepts from {filename}
in the classwork folder.

Instructions:
1. Read the corresponding classwork file first
2. Implement the solutions below
3. Run: python3 {filename}
4. Test your solutions
"""

# TODO: Implement based on the corresponding classwork file

# Your code here

if __name__ == "__main__":
    print("Homework: {base_name} - Implement the exercises above")
    pass
'''


def main():
    for phase, filenames in STUBS:
        phase_path = os.path.join(BASE, phase)
        cw_dir = os.path.join(phase_path, "classwork")
        hw_dir = os.path.join(phase_path, "homework")
        for fn in filenames:
            base_name = os.path.splitext(fn)[0]
            title = base_name.replace("_", " ").title()
            cw_path = os.path.join(cw_dir, fn)
            hw_path = os.path.join(hw_dir, fn)
            if not os.path.exists(cw_path):
                with open(cw_path, "w") as f:
                    f.write(CLASSWORK_TEMPLATE.format(title=title))
                print(f"  Created classwork/{fn}")
            if not os.path.exists(hw_path):
                with open(hw_path, "w") as f:
                    f.write(HOMEWORK_TEMPLATE.format(
                        filename=fn, base_name=base_name
                    ))
                print(f"  Created homework/{fn}")
    print("Done.")


if __name__ == "__main__":
    main()

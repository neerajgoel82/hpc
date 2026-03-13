"""
Homework: 26_json_files.py

Complete the exercises below based on the concepts from 26_json_files.py
in the classwork folder. Use the json module.

Instructions:
1. Read the corresponding classwork file first.
2. Implement each function below.
3. Run: python3 26_json_files.py
"""

import json

# Exercise 1: Convert a dict to a JSON string and return it
def dict_to_json_string(data):
    """Return json.dumps(data) with indent=2 for readability."""
    pass

# Exercise 2: Write a Python dict to a JSON file
def write_json_file(filepath, data):
    """Write data to filepath as JSON. Use json.dump()."""
    pass

# Exercise 3: Read a JSON file and return the parsed Python object
def read_json_file(filepath):
    """Return the Python object from the JSON file. Return None if file not found or invalid JSON."""
    pass

# Exercise 4: Create a dict with your profile (name, age, city, hobbies list) and save to profile.json
def save_my_profile():
    """Create a dict, save to profile.json, and print confirmation."""
    pass

# Exercise 5: Load JSON from file, modify one field, save back
def update_json_field(filepath, key, new_value):
    """Load filepath, set data[key] = new_value, write back. Return True on success."""
    pass


if __name__ == "__main__":
    print("=== Homework: 26_json_files ===\n")
    # Uncomment to test as you implement:
    # d = {"name": "Test", "count": 3}
    # print("dict_to_json_string:", dict_to_json_string(d))
    # write_json_file("test_data.json", d)
    # print("read_json_file:", read_json_file("test_data.json"))
    # save_my_profile()
    print("Implement the functions above, then uncomment the tests in main().")

"""
Exercise 26: Working with JSON

Learning Objectives:
- JSON format basics
- Reading JSON files
- Writing JSON files
- Converting Python objects to/from JSON
- Working with nested JSON
"""

import json

print("=== JSON BASICS ===\n")

# JSON is a text format for data exchange
# Similar to Python dictionaries

# Python dictionary
person = {
    "name": "Alice",
    "age": 30,
    "city": "Paris",
    "hobbies": ["reading", "cycling"]
}

# Convert to JSON string
json_string = json.dumps(person)
print(f"JSON string: {json_string}")
print(f"Type: {type(json_string)}")

# Pretty print with indentation
json_pretty = json.dumps(person, indent=2)
print(f"\nPretty JSON:\n{json_pretty}")

# TODO: Create dict and convert to JSON string


print("\n=== WRITING JSON FILES ===\n")

# Write JSON to file
data = {
    "users": [
        {"name": "Alice", "age": 30, "active": True},
        {"name": "Bob", "age": 25, "active": False}
    ],
    "total": 2
}

with open("users.json", "w") as file:
    json.dump(data, file, indent=2)
print("Data written to users.json")

# TODO: Write your own data to JSON file


print("\n=== READING JSON FILES ===\n")

# Read JSON from file
try:
    with open("users.json", "r") as file:
        loaded_data = json.load(file)
    print(f"Loaded data: {loaded_data}")
    print(f"First user: {loaded_data['users'][0]['name']}")
except FileNotFoundError:
    print("File not found")

# Parse JSON string
json_text = '{"name": "Charlie", "age": 35}'
parsed = json.loads(json_text)
print(f"Parsed: {parsed}")

# TODO: Read and parse JSON data


print("\n=== NESTED JSON ===\n")

# Complex nested structure
config = {
    "app": {
        "name": "MyApp",
        "version": "1.0.0",
        "settings": {
            "theme": "dark",
            "language": "en",
            "features": {
                "notifications": True,
                "auto_save": True
            }
        }
    },
    "users": [
        {
            "id": 1,
            "name": "Admin",
            "permissions": ["read", "write", "delete"]
        }
    ]
}

# Save nested structure
with open("config.json", "w") as file:
    json.dump(config, file, indent=2)

# Access nested values
print(f"App name: {config['app']['name']}")
print(f"Theme: {config['app']['settings']['theme']}")
print(f"Notifications: {config['app']['settings']['features']['notifications']}")

# TODO: Create and navigate complex nested JSON


print("\n=== JSON DATA TYPES ===\n")

# Supported types
data = {
    "string": "hello",
    "number": 42,
    "float": 3.14,
    "boolean": True,
    "null": None,
    "array": [1, 2, 3],
    "object": {"nested": "value"}
}

json_str = json.dumps(data)
print(json_str)

# Note: Python tuples become JSON arrays
python_data = {
    "tuple": (1, 2, 3),  # Will become array in JSON
    "set": {1, 2, 3}     # Will become array in JSON
}

# TODO: Understand JSON type conversions


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Save User Profile
# TODO: Create dict with your profile and save to JSON


# Exercise 2: Load and Modify
# TODO: Load JSON, modify data, save back


# Exercise 3: Config Manager
# TODO: Create config file reader/writer


# Exercise 4: Todo List
# TODO: JSON-based todo list (add, remove, mark complete)


# Exercise 5: Data Validation
# TODO: Load JSON and validate required fields


print("\n=== BONUS: API RESPONSE ===\n")

# Simulate API response
api_response = '''
{
    "status": "success",
    "data": {
        "weather": "sunny",
        "temperature": 25,
        "forecast": [
            {"day": "Monday", "temp": 26},
            {"day": "Tuesday", "temp": 24}
        ]
    }
}
'''

weather = json.loads(api_response)
print(f"Status: {weather['status']}")
print(f"Current: {weather['data']['temperature']}°C")
for day in weather['data']['forecast']:
    print(f"{day['day']}: {day['temp']}°C")

# TODO: Parse and work with JSON API responses


print("\nGreat work! Complete project_file_organizer.py")

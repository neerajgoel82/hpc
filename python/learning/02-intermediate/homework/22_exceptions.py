"""
Homework: 22_exceptions.py

Complete the exercises below based on the concepts from 22_exceptions.py
in the classwork folder. Use try/except/else/finally and raise.

Instructions:
1. Read the corresponding classwork file first.
2. Implement each function below.
3. Run: python3 22_exceptions.py
"""

# Exercise 1: Safely convert a string to int; return None on ValueError
def safe_int(s):
    """Return int(s), or None if conversion fails."""
    pass

# Exercise 2: Function with multiple exception handlers (e.g. divide a by b; handle ZeroDivisionError and TypeError)
def safe_divide(a, b):
    """Return a/b, or None and print an error message on ZeroDivisionError or TypeError."""
    pass

# Exercise 3: Get integer from user with a prompt; keep asking until valid input
def get_int_from_user(prompt="Enter an integer: "):
    """Use try/except in a loop. Return the integer when valid."""
    pass

# Exercise 4: Safely access list element at index; return None if IndexError
def safe_list_get(lst, index):
    """Return lst[index] or None if index out of range."""
    pass

# Exercise 5: Raise ValueError for invalid input
def set_age(age):
    """If age is not in 0-150, raise ValueError with a message. Otherwise return age."""
    pass


if __name__ == "__main__":
    print("=== Homework: 22_exceptions ===\n")
    # Uncomment to test as you implement:
    # print("safe_int('42') =", safe_int("42"))
    # print("safe_int('abc') =", safe_int("abc"))
    # print("safe_divide(10, 2) =", safe_divide(10, 2))
    # print("safe_divide(10, 0) =", safe_divide(10, 0))
    # print("safe_list_get([1,2,3], 1) =", safe_list_get([1, 2, 3], 1))
    # print("safe_list_get([1,2,3], 10) =", safe_list_get([1, 2, 3], 10))
    print("Implement the functions above, then uncomment the tests in main().")

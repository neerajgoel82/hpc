# Phase 3: Object-Oriented Programming (Weeks 7-8)

Now we enter the world of OOP - a fundamental paradigm in modern programming!

## What You'll Learn

- Classes and objects
- Attributes and methods
- Constructors (__init__)
- Inheritance and polymorphism
- Encapsulation
- Special methods (magic methods)
- Class vs instance variables
- Property decorators

## Week-by-Week Breakdown

### Week 7: OOP Basics
- `32_classes_objects.py` - Creating classes and objects
- `33_attributes_methods.py` - Instance variables and methods
- `34_constructors.py` - The __init__ method
- `35_class_variables.py` - Class vs instance variables
- `36_encapsulation.py` - Private variables and methods
- **Project:** `project_bank_account.py` - Banking system

### Week 8: Advanced OOP
- `37_inheritance.py` - Inheriting from parent classes
- `38_polymorphism.py` - Method overriding
- `39_magic_methods.py` - __str__, __repr__, __len__, etc.
- `40_properties.py` - @property decorator
- `41_composition.py` - Has-a relationships
- **Project:** `project_game_characters.py` - RPG character system

## Key Concepts

### Classes and Objects
```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        return f"{self.name} says Woof!"

my_dog = Dog("Buddy", 3)
print(my_dog.bark())
```

### Inheritance
```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"
```

## Why Learn OOP?

- **Organization**: Bundle related data and functions
- **Reusability**: Inherit and extend existing classes
- **Maintainability**: Changes in one place affect all instances
- **Real-world modeling**: Model real-world entities naturally

## Getting Started

```bash
cd phase3-oop
python3 32_classes_objects.py
```

## Prerequisites

Completed Phase 1 and 2:
- Strong understanding of functions
- Comfortable with dictionaries and data structures
- Understanding of scope

Ready to think in objects? Let's go!

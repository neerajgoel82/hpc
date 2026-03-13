"""
Exercise 37: Inheritance

Learning Objectives:
- Inherit from parent classes
- Override methods
- Use super()
- Multiple inheritance
- isinstance() and issubclass()
"""

print("=== BASIC INHERITANCE ===\n")

# Parent (base) class
class Animal:
    """Base class for all animals"""

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def speak(self):
        return "Some sound"

    def get_info(self):
        return f"{self.name}, age {self.age}"

# Child (derived) class
class Dog(Animal):
    """Dog inherits from Animal"""

    def speak(self):  # Override parent method
        return "Woof!"

class Cat(Animal):
    """Cat inherits from Animal"""

    def speak(self):
        return "Meow!"

# Use inherited classes
dog = Dog("Buddy", 3)
cat = Cat("Whiskers", 2)

print(dog.get_info())  # Inherited method
print(dog.speak())     # Overridden method
print(cat.get_info())
print(cat.speak())

# TODO: Create Vehicle parent and Car child class


print("\n=== SUPER() FUNCTION ===\n")

class Person:
    """Base person class"""

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"I'm {self.name}, {self.age} years old"

class Student(Person):
    """Student inherits from Person"""

    def __init__(self, name, age, student_id, major):
        super().__init__(name, age)  # Call parent constructor
        self.student_id = student_id
        self.major = major

    def introduce(self):
        parent_intro = super().introduce()  # Call parent method
        return f"{parent_intro}, studying {self.major}"

student = Student("Alice", 20, "S12345", "Computer Science")
print(student.introduce())

# TODO: Create Employee class inheriting from Person


print("\n=== MULTIPLE INHERITANCE ===\n")

class Flyable:
    """Mixin for flying ability"""

    def fly(self):
        return f"{self.name} is flying!"

class Swimmable:
    """Mixin for swimming ability"""

    def swim(self):
        return f"{self.name} is swimming!"

class Duck(Animal, Flyable, Swimmable):
    """Duck can fly and swim"""

    def speak(self):
        return "Quack!"

duck = Duck("Donald", 2)
print(duck.speak())
print(duck.fly())
print(duck.swim())
print(duck.get_info())

# TODO: Create class with multiple inheritance


print("\n=== ISINSTANCE() AND ISSUBCLASS() ===\n")

dog = Dog("Max", 4)
cat = Cat("Fluffy", 3)

# Check if object is instance of class
print(f"dog is Animal: {isinstance(dog, Animal)}")
print(f"dog is Dog: {isinstance(dog, Dog)}")
print(f"dog is Cat: {isinstance(dog, Cat)}")

# Check if class is subclass
print(f"Dog is subclass of Animal: {issubclass(Dog, Animal)}")
print(f"Cat is subclass of Animal: {issubclass(Cat, Animal)}")
print(f"Dog is subclass of Cat: {issubclass(Dog, Cat)}")

# TODO: Practice with isinstance() and issubclass()


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Shape Hierarchy
# TODO: Create Shape parent class
# Children: Rectangle, Circle, Triangle
# Each with area() and perimeter() methods


# Exercise 2: Employee Types
# TODO: Create Employee parent
# Children: Manager, Developer, Designer
# Different salary calculations


# Exercise 3: Bank Accounts
# TODO: Create Account parent
# Children: SavingsAccount, CheckingAccount
# Different interest rates and fees


# Exercise 4: Media Library
# TODO: Create Media parent
# Children: Book, Movie, Music
# Different properties and play/read methods


# Exercise 5: Vehicle Fleet
# TODO: Create Vehicle parent
# Children: Car, Truck, Motorcycle
# Different fuel efficiency and capacity


print("\nExcellent! Next: 38_polymorphism.py")

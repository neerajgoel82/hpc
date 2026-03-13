/*
 * File: 04_structures.c
 * Topic: Structures in C
 *
 * Structures allow you to group related data of different types together.
 * They're fundamental for creating complex data types.
 *
 * Key Concepts:
 * - Structure definition
 * - Structure variables
 * - Accessing members
 * - Nested structures
 * - Array of structures
 * - Structures and functions
 * - typedef
 */

#include <stdio.h>
#include <string.h>

// Structure definition
struct Student {
    int id;
    char name[50];
    float gpa;
};

// Structure with nested structure
struct Date {
    int day;
    int month;
    int year;
};

struct Person {
    char name[50];
    int age;
    struct Date birthdate;  // Nested structure
};

// Using typedef
typedef struct {
    float x;
    float y;
} Point;

// Function prototypes
void printStudent(struct Student s);
void printStudentByPointer(struct Student *s);
void modifyStudent(struct Student *s);
float calculateDistance(Point p1, Point p2);

int main() {
    // Creating structure variables
    printf("=== Basic Structure ===\n");
    struct Student student1;

    // Assigning values to members
    student1.id = 1001;
    strcpy(student1.name, "Alice");
    student1.gpa = 3.8;

    // Accessing and printing members
    printf("ID: %d\n", student1.id);
    printf("Name: %s\n", student1.name);
    printf("GPA: %.2f\n", student1.gpa);

    // Initialization during declaration
    printf("\n=== Structure Initialization ===\n");
    struct Student student2 = {1002, "Bob", 3.5};
    printf("Student: %s (ID: %d, GPA: %.2f)\n",
           student2.name, student2.id, student2.gpa);

    // Designated initializers
    struct Student student3 = {
        .name = "Charlie",
        .id = 1003,
        .gpa = 3.9
    };
    printStudent(student3);

    // Nested structures
    printf("\n=== Nested Structures ===\n");
    struct Person person = {
        .name = "David",
        .age = 25,
        .birthdate = {15, 6, 1998}
    };

    printf("Name: %s\n", person.name);
    printf("Age: %d\n", person.age);
    printf("Birthdate: %02d/%02d/%d\n",
           person.birthdate.day,
           person.birthdate.month,
           person.birthdate.year);

    // Array of structures
    printf("\n=== Array of Structures ===\n");
    struct Student class[3] = {
        {1, "Alice", 3.8},
        {2, "Bob", 3.5},
        {3, "Charlie", 3.9}
    };

    printf("Class roster:\n");
    for (int i = 0; i < 3; i++) {
        printf("%d. %s (GPA: %.2f)\n",
               class[i].id, class[i].name, class[i].gpa);
    }

    // Pointers to structures
    printf("\n=== Pointers to Structures ===\n");
    struct Student student4 = {1004, "Eve", 4.0};
    struct Student *ptr = &student4;

    // Accessing members through pointer (two ways)
    printf("Using (*ptr).member: %s\n", (*ptr).name);
    printf("Using ptr->member: %s\n", ptr->name);
    printf("GPA: %.2f\n", ptr->gpa);

    // Passing structure to function
    printf("\n=== Passing to Functions ===\n");
    printStudent(student4);
    printStudentByPointer(&student4);

    printf("Before modification:\n");
    printStudent(student4);
    modifyStudent(&student4);
    printf("After modification:\n");
    printStudent(student4);

    // Using typedef structures
    printf("\n=== typedef Structures ===\n");
    Point p1 = {0.0, 0.0};
    Point p2 = {3.0, 4.0};

    printf("Point 1: (%.1f, %.1f)\n", p1.x, p1.y);
    printf("Point 2: (%.1f, %.1f)\n", p2.x, p2.y);
    printf("Distance: %.2f\n", calculateDistance(p1, p2));

    // Structure assignment
    printf("\n=== Structure Assignment ===\n");
    struct Student original = {1005, "Frank", 3.7};
    struct Student copy = original;  // Copies all members

    printf("Original: %s\n", original.name);
    printf("Copy: %s\n", copy.name);

    strcpy(copy.name, "George");
    printf("After modifying copy:\n");
    printf("Original: %s (unchanged)\n", original.name);
    printf("Copy: %s\n", copy.name);

    // sizeof structure
    printf("\n=== Structure Size ===\n");
    printf("sizeof(struct Student): %zu bytes\n", sizeof(struct Student));
    printf("sizeof(struct Person): %zu bytes\n", sizeof(struct Person));
    printf("sizeof(Point): %zu bytes\n", sizeof(Point));

    return 0;
}

// Pass structure by value (copy)
void printStudent(struct Student s) {
    printf("Student: %s (ID: %d, GPA: %.2f)\n", s.name, s.id, s.gpa);
}

// Pass structure by pointer (more efficient)
void printStudentByPointer(struct Student *s) {
    printf("Student: %s (ID: %d, GPA: %.2f)\n", s->name, s->id, s->gpa);
}

// Modify structure through pointer
void modifyStudent(struct Student *s) {
    s->gpa += 0.1;
    strcat(s->name, " (Honor Roll)");
}

// Calculate distance between two points
float calculateDistance(Point p1, Point p2) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    return dx * dx + dy * dy;  // Simplified: returns squared distance
}

/*
 * STRUCTURE SYNTAX:
 *
 * Definition:
 * struct StructureName {
 *     type member1;
 *     type member2;
 * };
 *
 * Declaration:
 * struct StructureName variable;
 *
 * Access:
 * variable.member       (for structure variables)
 * pointer->member       (for structure pointers)
 * (*pointer).member     (alternative for pointers)
 *
 * TYPEDEF:
 * typedef struct {
 *     type member;
 * } TypeName;
 *
 * Now you can use: TypeName variable;
 * Instead of: struct StructureName variable;
 *
 * IMPORTANT NOTES:
 * - Structures group related data
 * - Pass large structures by pointer for efficiency
 * - Use -> to access members through pointers
 * - Structure assignment copies all members
 * - sizeof(struct) may include padding for alignment
 *
 * WHEN TO USE STRUCTURES:
 * - Representing real-world entities (Person, Car, etc.)
 * - Grouping related data (coordinates, date/time, etc.)
 * - Return multiple values from function
 * - Building complex data structures (linked lists, trees)
 *
 * EXERCISES:
 * 1. Create a Book structure with title, author, price
 * 2. Write a function to find student with highest GPA
 * 3. Create a Rectangle structure and function to calculate area
 * 4. Implement a simple contact management system
 * 5. Create a Time structure and function to compare two times
 * 6. Build an Employee database with array of structures
 * 7. Create a Circle structure with center point and radius
 * 8. Implement a simple banking system with Account structure
 */
